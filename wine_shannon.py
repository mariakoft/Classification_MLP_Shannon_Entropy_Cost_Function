# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:17:20 2018

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 03:19:14 2018

@author: gb
"""


import numpy as np
import keras as k
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


start = time.time()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe1 = pandas.read_csv("wine.csv", header=None)
dataset = dataframe1.values
X = dataset[:,1:].astype(float)
X = preprocessing.scale(X) #OPTIONAL to scale data
Y = dataset[:,0]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X, trainX, dummy_y, dummy_trainy = train_test_split(X, dummy_y, stratify = dummy_y)

#batch=number of train samples
def shannon_entropy(batch):
    def loss(y_true, y_pred):
        def g_k(x):
            pi = tf.convert_to_tensor(np.pi, dtype=tf.float32)
            return 1./ K.pow(2. * pi, int(x.get_shape()[-1])/2) * K.exp(-0.5 * K.square(x))
        def f_e(l):
            E = tf.map_fn(lambda x: 1./((batch)*K.var(l))*K.sum(g_k((x-l)/K.var(l)), axis=-1), l)
            return E
        errors = y_true - y_pred 
        errors = K.square(errors)
        print(errors.get_shape()[-1])
        print(errors)
        H = - 1./(batch)*K.sum(K.log(f_e(errors)), axis=-1)
        return H
    return loss

# define baseline model
model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
counts = np.unique(encoded_Y, return_counts=True)
model.add(Dense(len(counts[0]), activation='softmax'))

adam= k.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile model
model.compile(loss=shannon_entropy(batch = len(X)), optimizer=adam, metrics=['accuracy'])
history = model.fit(X, dummy_y, validation_split=0.15, epochs=500, batch_size=len(X),
                     shuffle = True)
scores=model.evaluate(trainX, dummy_trainy, verbose=0)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Prediction
predictions = model.predict_classes(np.array(trainX))

for i in range(len(trainX)):
	print("X=%s, Predicted=%s" % (predictions[i], dummy_trainy[i]))
    
    
# Calculate predictions accuracy
    
# one hot encode
encoded = to_categorical(predictions)
print(encoded)

temp=0
for i in range(len(predictions)):
    if (dummy_trainy[i] == encoded[i]).all():
        temp=temp+1
    else:
        temp=temp
prediction_accuracy=temp/len(dummy_trainy)
print(prediction_accuracy)

end = time.time()
print((end-start)/60, "min")

