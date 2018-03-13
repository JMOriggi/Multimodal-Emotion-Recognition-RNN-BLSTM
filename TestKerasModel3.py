#LSTM INPUT ARRAY SHAPE: 3D tensor with shape (batch_size, timesteps, input_dim)
##Batchsize: numero di sample da considerare come chunk, quindi considero ad esmepio i primi 10 poi i successivi 10 riallenando la rete ogni volta.
#MODEL DECLARATION : LSTM(hidden_nodes, input_shape=(timesteps, input_dim)))
##Hidden_nodes = This is the number of neurons of the LSTM. If you have a higher number, the network gets more powerful.
##Timesteps = the number of timesteps you want to consider. E.g. if you want to classify a sentence, this would be the number of words in a sentence.
##Input_dim = the dimensions of your features/embeddings. E.g. a vector representation of the words in the sentence
#INPUT MATRIX SHAPE FOR KERAS: [A B C ...]--> struttura 1 input con valori ABC...
#                                [[A B C ..]]--> struttura 1 sample
#                                [ [[a b c]] [[d e f]] ]--> struttura array X con 2 sample 
#OUTPUT MATRIX SHAPE FOR KERAS: [ [1] [2] [3] ]--> output shape with 3 sample output.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

#PREPARE TRAINING DATA
def get_train():
    data = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])
    labels = [[2],[3],[4],[5],[6]]
    X = data
    y = np.asarray(labels)
    print('X: ', X)
    print('Y: ', y)
    X = X.reshape((len(X),1, 2))
    return X, y

#DEFINE MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(1,2)))
model.add(Dense(1, activation='linear'))

#COMPILE MODEL
model.compile(loss='mse', optimizer='adam')

#GET DATA FOR TRAINING
X,y = get_train()
print('X: ', X)
print('Y: ', y)

#Train MODEL
model.fit(X, y, epochs=5000, shuffle=False, verbose=2)

#SAVE MODEL AND WEIGHTS AFTER TRAINING
model.save('lstm_model.h5')


