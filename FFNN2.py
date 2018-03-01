import tensorflow as tf
import numpy as np


#Input data
#generate a matrix of random number with 2 row (2 inputs) and 100 values
x_data = np.float32(np.random.rand(2,5))
print('Input',x_data)

#Output data, product of 2 array. 1 neurone con 2 ingressi
y_data = np.dot([0.1,0.2], x_data) + 0.3 
#l output della rete si avvicinera il piu possibile a questa y_data
#visto che l'output e il prodotto tra un piccolo array di 2 weight e un array grande di x_data input
#i miei coeficienti da allenare saranno solo 2 e dovranno dopo l'allenamento avvicinarsi il piu possibile a 0.1 e 0.2
#QUESTA E UNA SIMULAZIONE POICHE LA FORMULA DELL'OUTPUT NON DOVREBBE ESSERE CONOSCIUTA
print('Output Basic',y_data)

#In tensor flow we use tensor variables, the framework consider data as tensor (a 3d matrix)
#Those tensors flows to the output
b = tf.Variable(tf.zeros(1))#our bias
#tf.random_uniform([minvalue,maxvalue],mean,stddev
W = tf.Variable(tf.random_uniform([1,2],-1,1))#W as weight or synapses initialized as random number
y = tf.matmul(W,x_data) + b #real output data use a multiply function to multiply input with weight and add bias

#Gradient descent time! to minimize the error
error = tf.reduce_mean(tf.square(y - y_data))#mean square error
optimizer = tf.train.GradientDescentOptimizer(0.5)#0.5 is the learning rate, the error step
train = optimizer.minimize(error) #we want to minimize our error 


#initialize all variables in tensor flow
init = tf.global_variables_initializer()
#launch the graph
sess = tf.Session()#Create a session, that is an event where we compute what we are trying to compute
sess.run(init)#the session will be run with all the tf variables

#print('Output Before run',sess.run(y))#Grossa matrice piena di valori random poiche W e settata a merda

#train -- fit the plane
for step in range(0,200):
    sess.run(train)
    #print every 20 step
    if step % 20 == 0:
        print('STEP: ',step,', error: ',sess.run(error),', weights: ',sess.run(W),', bias: ',sess.run(b))

print('Output After run',sess.run(y))
    
    

