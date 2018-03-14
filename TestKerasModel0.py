from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(type(x_train))
print(x_train)
print(x_train[0])
print('Output')
print(type(y_train))
print(y_train)
print(y_train[0])
print('x test')
print(type(x_test))
print(x_test)
print(x_test[0])
print('y test')
print(type(y_test))
print(y_test)
print(y_test[0])

'''print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

#SET THE MODEL: LSTM(output, inputShape=type of input matrix very important)
model = Sequential()
model.add(LSTM(1, input_shape=(1, 1024)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size,epochs=4,validation_data=[x_test, y_test])'''



