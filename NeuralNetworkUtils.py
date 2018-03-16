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
        X = np.full((len(Input), 80,len(Input[0][0])), 0)
        
        #y = np.full((len(X), 1), output)
        Y = np.asarray(output)
        Y = Y.reshape((len(Y), 1))
        
        #y<80 problem of the lenght of the audio
        i = 0
        while i<len(Input):
            y = 0
            while y<80:
                #print('X: ', X[i][y])
                #print('len X: ', len(X[i][y]))
                #print('len input: ', len(Input[i][y]))
                X[i][y] = Input[i][y]
                #print('X: ', X[i][y])
                y+=1
            i+=1
        
        return X, Y
    
    #GET DATA FOR TRAINING
    X, Y = get_train()
    print('len X1: ', len(X))
    print('len X2: ', len(X[0]))
    print('len X3: ', len(X[0][0]))
    print('X: ', X)
    print('len Y1: ', len(Y))
    print('len Y2: ', len(Y[0]))
    print('Y: ', Y)
    
    #DEFINE MODEL
    model = Sequential()
    #model.add(LSTM(10, input_shape=(len(X[0]),len(X[0][0])), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(LSTM(64, input_shape=(len(X[0]),len(X[0][0]))))
    #model.add(Dense(1,activation='softmax'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, Y, epochs=500, verbose=2)
    
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
    
    