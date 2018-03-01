#FEED FOWARD NERUAL NETWORK

import numpy as np

#NEURONE: FUNZIONE SIGMOIDE CHE SARA RICHIAMATA PER OGNI NEURONE
#la sigmoide trasforma un numero in una probabilita (tra 0 ed 1)
#Se passiamo deriv==True la derivata sara calcolata, che mi serve quando faccio
#back propagation per calcolarmi l errore, durante il Training della rete
def nonlin(x, deriv=False):
    if(deriv==True):
        return(x*(1-x))
    if(deriv==False):
        return 1/(1+np.exp(-x))


#input data TRAINING
x= np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

#output data TRAINING
y=np.array([[0],
           [1],
           [1],
           [0]])

#seed: without hat it will not be reproduceble
np.random.seed(1)

#SYNAPSES: sono i collegamenti che corrispondono ai miei coeficienti WEIGHT generati random
#Saranno delle matrici (node,outpout) a cui sottrago il bias
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

#TRAINING: sara un ciclo for
for j in range(60000):
    #layers
    l0 = x #input
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
        
    #BACKPROPAGATION
    #ERROR: e la differenza tra il mio output settato y e quello che ottengo dal result layer l2 
    #DELTA: e l'updating part uso questa varianza per modificare il mio coeficiente
    l2_error = y-l2    
    l2_delta=l2_error*nonlin(l2, deriv=True)
    
    l1_error=l2_delta.dot(syn1.T)   
    l1_delta=l1_error*nonlin(l1,deriv=True)

    #JUST PRINT THE ERROR: mean of the absolute value of the error to print it
    #lo stampo ogni 10000 step
    if(j%10000) == 0:
        print ('Error:'+str(np.mean(np.abs(l2_error))))

    #updat our synpases
    syn0+=l0.T.dot(l1_delta)
    syn1+=l1.T.dot(l2_delta)   
  
#risultato che dovrebbe avvicinarsi il piu possibile all'output d riferimento
#si tratta di probabilita    
print ('output after training')
print (l2)


