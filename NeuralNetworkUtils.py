import tensorflow as tf
import numpy as np

    
def FFModel(In):
    print('****Start of method FFModel')
    print('****End of method FFModel')

    
def RNNModel(In):
    print('****Start of method RNNModel')
    print('****End of method RNNModel')


def FFNNModelTraining(x_data, yTrainData):
    print('****Start of method FFNNModel')
    
    #STARTING SITUATION
    print('Input: ', x_data)
    print('Output training data: ', yTrainData)
    
    #NETWORK DEFINITION: we define the biases, weights, and outputs of our synapses as tensors variable
    b = tf.Variable(tf.zeros(1))#a 1 value matrix initialized with 0
    W = tf.Variable(tf.random_uniform([1,2],-1,1))#W as weight or synapses matrix of 1 row and 2 collumns (collumns correspond to the number of input present) initialized as random number
    y = tf.matmul(W,x_data) + b #Output data: input times weight add a bias
    
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
    
    