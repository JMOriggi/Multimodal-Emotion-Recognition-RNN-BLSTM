#The input matrix: rows==number of differents inputs / columns == lenght of each inputs.
#The weight matrix: rows == number of neurons / columns == number of entry for each neurons (corresponding to the number of inputs). 


import tensorflow as tf
import numpy as np


#INPUT DATA PREPARATION
#generate a matrix of random number with 2 row (2 inputs) and 100 values (length of each input)
x_data = np.float32(np.random.rand(2,10))
print('Input: ',x_data)

#TRAINING OUTPUT DATA: simulating with 1 neurone of 2 inputs
#Simulation with known weight, usually we don't know the weights we only have the output and input for training.
#In this case we simulate the output, and at the end of the training we must have the closest output and weights to the choosen one.
#Weight Dovranno dopo l'allenamento avvicinarsi il piu possibile a 0.1 e 0.2
w = [[0.1, 0.2]]
y_data = np.dot([0.1,0.2], x_data) + 0.3 
print('Output GOAL: ',y_data)
print('Weights GOAL: ',w)

#NETWORK DEFINITION: we define the biases, weights, and outputs of our synapses as tensors variable
b = tf.Variable(tf.zeros(1))#a 1 value matrix initialized with 0
W = tf.Variable(tf.random_uniform([1,2],-1,1))#W as weight or synapses matrix of 1 row and 2 collumns (collumns correspond to the number of input present) initialized as random number
y = tf.matmul(W,x_data) + b #Output data: input times weight add a bias

#TRAINING SETTINGS: Gradient descent time! to minimize the error
error = tf.reduce_mean(tf.square(y - y_data))#Define the error function as the mean square error
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
    
    

