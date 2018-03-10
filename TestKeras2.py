from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array

#PREPARE TRAINING DATA
def get_train():
    data = array(range(0,25))
    labels = array([range(1,26)],[range(2,27)])
    X = data
    y = labels
    X = X.reshape((len(X), 1, 1))
    return X, y

#DEFINE MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
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





