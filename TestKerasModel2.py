
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

#PREPARE TRAINING DATA
def get_train():
    data = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])
    labels = [[1,2],[2,3],[3,4],[4,5],[5,6]]
    X = data
    y = np.asarray(labels)
    X = X.reshape((len(X),1, 2))
    return X, y

#DEFINE MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(1,2)))
model.add(Dense(2, activation='linear'))

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





