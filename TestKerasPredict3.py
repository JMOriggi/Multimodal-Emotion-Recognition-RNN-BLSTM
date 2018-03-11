import numpy as np
from keras.models import load_model

#LOAD MODEL FROM FILE
model = load_model('lstm_model.h5')

#PREDICTION
#test = array([5])
test = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])
test = test.reshape((len(test), 1, 2))

#yhat = model.predict(X, verbose=0)
yhat = model.predict(test, verbose=0)

print('Result: ',yhat)