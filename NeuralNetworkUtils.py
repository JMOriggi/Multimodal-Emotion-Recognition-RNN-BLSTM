import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

    
def FFModel(In):
    print('****Start of method FFModel')
    print('****End of method FFModel')

    
def RNNModel(Input, output):
    print('****Start of method RNNModel')
    
    #PREPARE TRAINING DATA
    def get_train():
        X = np.full((len(Input), 1,len(Input[0])), 0)
        #X = np.asarray(Input)
        #X = np.delete(X, (len(X)-1), axis=0)
        #print('len before: ', len(Input))
        #print('len after: ', len(X))
        #X = X.reshape((len(X),1, len(Input[0])))
        print('X: ', X)
        
        y = np.full((len(X), 1), output)
        print('Y: ', y)
        
        #Not considering last frame beacause lenght not costant for the last frame
        i = 0
        while i<len(Input)-1:
            print('X: ', X[i][0])
            X[i][0] = Input[i]
            print('X: ', X[i][0])
            i+=1
        print('X: ', X)
        
        return X, y
    
    #GET DATA FOR TRAINING
    X, y = get_train()
    print('X: ', X)
    print('Y: ', y)
    
    #DEFINE MODEL
    model = Sequential()
    model.add(LSTM(10, return_sequences=False, input_shape=(1,len(Input[0]))))
    model.add(Dense(1, activation='linear'))
    
    #COMPILE MODEL
    model.compile(loss='mse', optimizer='adam')
    
    #Train MODEL
    model.fit(X, y, epochs=100, shuffle=False, verbose=2)
    
    #SAVE MODEL AND WEIGHTS AFTER TRAINING
    model.save('RNN_Model_saved.h5')
    
    #TEST MODEL
    predictFromSavedModel(X, 'RNN_Model_saved.h5')
    
    print('****End of method RNNModel')
    



#INPUT: must be an array where each row is an input; 
def FFNNModel(x_data1, yTrainData):
    print('****Start of method FFNNModel')
    
    #TRASFORM INPUT TO ACCEPTED FORM: input alla rete deve avere forma [[.. .. ..]] quindi una matrice 1xN
    N = len(x_data1[0])
    x_data = x_data1[0].reshape((1,N))
    
    #STARTING SITUATION
    print('Input: ', x_data)
    print('Type input: ',type(x_data))
    print('Shape input: ',np.shape(x_data))
    print('Output training data: ', yTrainData)
    
    #NETWORK DEFINITION: we define the biases, weights, and outputs of our synapses as tensors variable
    b = tf.Variable(tf.zeros(1))#a 1 value matrix initialized with 0
    W = tf.Variable(tf.random_uniform([1,1],-1,1))#W as weight or synapses matrix of 1 row and 2 collumns (collumns correspond to the number of input present) initialized as random number
    y = tf.matmul(W,np.float32(x_data)) + b #Output data: input times weight add a bias
    
    #TRAINING SETTINGS: Gradient descent time! to minimize the error
    error = tf.reduce_mean(tf.square(y - yTrainData))#Define the error function as the mean square error
    optimizer = tf.train.GradientDescentOptimizer(0.5)#Associate the optimizer functionas a gradient descent function, 0.5 is the learning rate, the error step
    train = optimizer.minimize(error) #We want to minimize the error with the prevously define function, we define another funciton to do that
    
    #INITIALIZE SESSION FOR TENSORFLOW
    init = tf.global_variables_initializer()
    sess = tf.Session()#Create a session, that is an event where we compute what we are trying to compute
    sess.run(init)#the session will be run with all the tf variables
    
    #TRAINING SESSION ROUTINE: 200 iterations of training
    print('Output Before run',sess.run(y))
    for step in range(0,300):
        sess.run(train)
        if step % 20 == 0: #print every 20 step
            print('STEP: ',step,', error: ',sess.run(error),', weights: ',sess.run(W),', bias: ',sess.run(b))
    print('Output After run',sess.run(y))

    print('****Start of method FFNNModel')
 
    
def predictFromSavedModel(test, fileName):
    print('****Start of method SaveWeights')
    
    #LOAD MODEL FROM FILE
    model = load_model(fileName)
    
    #yhat = model.predict(X, verbose=0)
    yhat = model.predict(test, verbose=0)
    
    print('Result: ',yhat)
    
    print('****End of method SaveWeights')        
    
    