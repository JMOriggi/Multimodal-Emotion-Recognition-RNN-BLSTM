from numpy import array
from keras.models import load_model

#LOAD MODEL FROM FILE
model = load_model('lstm_model.h5')

#PREDICTION
#test = array([5])
test = array(range(0,25))
test = test.reshape((len(test), 1, 1))

#yhat = model.predict(X, verbose=0)
yhat = model.predict(test, verbose=0)

print('Result: ',yhat)