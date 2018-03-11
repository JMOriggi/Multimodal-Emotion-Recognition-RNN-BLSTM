import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

    
def FFModel(In):
    print('****Start of method FFModel')
    print('****End of method FFModel')

    
def RNNModel(input, output):
    print('****Start of method RNNModel')
    
    #PREPARE TRAINING DATA
    def get_train():
        X = np.asarray(input)
        y = np.asarray(output)
        print('X: ', X)
        print('Y: ', y)
        X = X.reshape((len(input[0]),len(X), len(input[0])))
        return X, y
    
    #DEFINE MODEL
    model = Sequential()
    model.add(LSTM(10, input_shape=(1,2)))
    model.add(Dense(1, activation='linear'))
    
    #COMPILE MODEL
    model.compile(loss='mse', optimizer='adam')
    
    #GET DATA FOR TRAINING
    X, y = get_train()
    print('X: ', X)
    print('Y: ', y)
    
    #Train MODEL
    model.fit(X, y, epochs=5000, shuffle=False, verbose=2)
    
    #SAVE MODEL AND WEIGHTS AFTER TRAINING
    model.save('lstm_model.h5')
    
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
 
    
def SaveWeights(Model):
    print('****Start of method SaveWeights')
    print('****End of method SaveWeights')        
    
    