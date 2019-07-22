import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter

sentiment_data = pd.read_csv('data.csv')  
sentiment_data.columns =['Class', 'Data']

unlabeld_data = pd.read_csv('unlabeld_data.txt') 
unlabeld_data.columns = ['Data']

#############################
#      Preprocessing        #
#############################

sentiment_data.head()
unlabeld_data.head()

# 1. Shuffle dataframe
########################
# The dataset is well sorted. First, we have half of data samples that are positive and then half of them negative.
#If we separate the dataset to training and testing parts like this, we will have most of the data (if not all) from one class.
#To prevent that from happening, we will shuffle the dataset first.

from sklearn.utils import shuffle
sentiment_data = shuffle(sentiment_data)
unlabeld_data = shuffle(unlabeld_data)

sentiment_data.head()

# #### 2 Split to labels and reviews
############################################# 
# In this step we need to create separated variables that will hold labels (positive or negative) and reviews.

labels = sentiment_data.iloc[:, 0].values
reviews = sentiment_data.iloc[:, 1].values
unlabeled_reviews = unlabeld_data.iloc[:,0].values


# 3 Clean data from punctuation
##############################################
# The punctuation shouldn't affect our prediction so we will delete all punctuation from reviews.

reviews_processed = []
unlabeled_processed = [] 
for review in reviews:
    review_cool_one = ''.join([char for char in review if char not in punctuation])
    reviews_processed.append(review_cool_one)
    
for review in unlabeled_reviews:
    review_cool_one = ''.join([char for char in review if char not in punctuation])
    unlabeled_processed.append(review_cool_one)


# 4 Creating vocabulary, coverting all characters to lower case and spliting each review into words
########################################################################################################################################
# In this step we are creating vocabulary which will be created by using function Counter.
#Also in this step we will lower all characters in the dataset, we can do this as well because lower/upper case character won't affect prediction results.
#Lastly, we will split each review to separate words.



word_reviews = []
word_unlabeled = []
all_words = []
for review in reviews_processed:
    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

for review in unlabeled_processed:
    word_unlabeled.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())
    
counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)


# 5 Creating vocab_to_int dictionary which will map word with a number
##########################################################################################


vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}


# 6 Using vocab_to_int to transform each review to vector of numbers
##########################################################################################


reviews_to_ints = []
for review in word_reviews:
    reviews_to_ints.append([vocab_to_int[word] for word in review])


unlabeled_to_ints = []

for review in word_unlabeled:
    unlabeled_to_ints.append([vocab_to_int[word] for word in review])


# #### Step 1.7 Check if we have some 0 length reviews.
##########################################################################################

reviews_lens = Counter([len(x) for x in reviews_to_ints])
print('Zero-length {}'.format(reviews_lens[0]))
print("Max review length {}".format(max(reviews_lens)))


# 8 Creating word vectors
# 
#     1. Define sequence length. (250 in this case)
#     2. Each review shorted then this sequence will be padded (at the beginning) with zeros
#     3. Each review longer than the sequence length will be shortened.


seq_len = 250

features = np.zeros((len(reviews_to_ints), seq_len), dtype=int)
for i, review in enumerate(reviews_to_ints):
    features[i, -len(review):] = np.array(review)[:seq_len]
    
features_test = np.zeros((len(unlabeled_to_ints), seq_len), dtype=int)
for i, review in enumerate(unlabeled_to_ints):
    features_test[i, -len(review):] = np.array(review)[:seq_len]


# 9 Split into training and testing parts


X_train = features[:6400]
y_train = labels[:6400]

X_test = features[6400:]
y_test = labels[6400:]

X_unlabeled = features_test

print('X_trian shape {}'.format(X_train.shape))
print('X_unlabeled shape {}'.format(X_unlabeled.shape))


# ### Done with preprocessing pipeline

#########################################
#           Defining RNN(LSTM)          #
#########################################


hidden_layer_size = 512 # no of nodes LSTM cells
number_of_layers = 1 # no of RNN layers
batch_size = 100 # no of comments we feed at once
learning_rate = 0.001 # learning rate
number_of_words = len(vocab_to_int) + 1 # unique words we have in vocab (+1  is used for 0 - padding)
dropout_rate = 0.8 
embed_size = 300 # word embedings length
epochs = 10 # no of epochs


tf.reset_default_graph() #Clean the graph


# Step 2.1 Define placeholders


inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
targets = tf.placeholder(tf.int32, [None, None], name='targets')


# Step 2.2 Define embeding layer


word_embedings = tf.Variable(tf.random_uniform((number_of_words, embed_size), -1, 1))
embed = tf.nn.embedding_lookup(word_embedings, inputs)


# Step 2.3 Define hidden layer and Dynamic RNN


hidden_layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
hidden_layer = tf.contrib.rnn.DropoutWrapper(hidden_layer, dropout_rate)

cell = tf.contrib.rnn.MultiRNNCell([hidden_layer]*number_of_layers)
init_state = cell.zero_state(batch_size, tf.float32)


outputs, states = tf.nn.dynamic_rnn(cell, embed, initial_state=init_state)


# Step 2.4 Get the prediction for each review 
 
# From the last step of our network we get output and use it as a prediction. Than we use that result and compare it with real sentiment for that review.


prediction = tf.layers.dense(outputs[:, -1], 1, activation=tf.sigmoid)
cost = tf.losses.mean_squared_error(targets, prediction)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Step 2.5 accuracy

currect_pred = tf.equal(tf.cast(tf.round(prediction), tf.int32), targets)
accuracy = tf.reduce_mean(tf.cast(currect_pred, tf.float32))

#################
#   Training    #
#################

session = tf.Session()

session.run(tf.global_variables_initializer())

for i in range(epochs):
    training_accurcy = []
    ii = 0
    epoch_loss = []
    length = len(X_train)
    while ii + batch_size <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size].reshape(-1, 1)
        
        a, o, _ = session.run([accuracy, cost, optimizer], feed_dict={inputs:X_batch, targets:y_batch})

        training_accurcy.append(a)
        epoch_loss.append(o)
        ii += batch_size
        print ('Batch: {}/{}'.format(ii,length), end='  ')
    print('Epoch: {}/{}'.format(i, epochs), ' | Current loss: {}'.format(np.mean(epoch_loss)),
          ' | Training accuracy: {:.4f}'.format(np.mean(training_accurcy)*100))


test_accuracy = []
length = len(X_test)
ii = 0
while ii + batch_size <= len(X_test):
    X_batch = X_test[ii:ii+batch_size]
    y_batch = y_test[ii:ii+batch_size].reshape(-1, 1)

    a = session.run([accuracy], feed_dict={inputs:X_batch, targets:y_batch})
    
    test_accuracy.append(a)
    ii += batch_size
    print ('Batch: {}/{}'.format(ii,length), end='  ')
print("Test accuracy is {:.4f}%".format(np.mean(test_accuracy)*100))

#####################################
#   Testing on the unlabeld data    #
#####################################


predictions_unlabeled = []
ii = 0
while ii + batch_size <= len(X_unlabeled):
    if ii + batch_size > len(X_unlabeled):
        batch_size = len(X_unlabeled) - ii
    X_batch = X_unlabeled[ii:ii+batch_size]
    y_batch = X_unlabeled[ii:ii+batch_size].reshape(-1, 1)

    pred = session.run([prediction], feed_dict={inputs:X_batch, targets:y_batch})
    
    predictions_unlabeled.append(pred)
    ii += batch_size


pred_real = []
for i in range(len(predictions_unlabeled)):
    for ii in range(len(predictions_unlabeled[i][0])):
        if predictions_unlabeled[i][0][ii][0] >= 0.5:
            pred_real.append(1)
        else:
            pred_real.append(0)


np.savetxt('predictions.txt', pred_real)

new_dataframe = unlabeld_data[:len(pred_real)]


new_dataframe['Classes'] = pred_real


print(new_dataframe)

