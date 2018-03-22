import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

    
def RNNModel(modelRNN, Input, output):
    print('****Start of method RNNModel')
    
    #PREPARE INPUT AND OUTPUT
    X = np.asarray(Input) 
    Y = np.asarray(output)
    Y = Y.reshape((len(Y), 7))
    '''print('len X1: ', len(X))
    print('len X2: ', len(X[0]))
    print('len X3: ', len(X[0][0]))
    print('X: ', X)
    print('len Y1: ', len(Y))
    print('len Y2: ', len(Y[0]))
    print('Y: ', Y)'''
    
    #DEFINE MODEL: if model do not exist create it otherwise use the given one.
    if modelRNN == '':
        model = Sequential()
        model.add(LSTM(64, input_shape=(None,len(X[0][0])), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        #model.add(LSTM(64, input_shape=(None,len(X[0][0])), return_sequences=False))
        #model.add(LSTM(64, input_shape=(len(X[0]),len(X[0][0])), return_sequences=False))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='mse', optimizer='adam')
        #model.compile(loss='categorical_crossentropy', optimizer='adam')
    else:
        model = modelRNN 
        
    model.fit(X, Y, epochs=10, verbose=0)
    
    print('****End of method RNNModel\n')
    return model
    
    
def predictFromSavedModel(inputTest, fileName):
    print('****Start of method predictFromSavedModel')
    
    #LOAD MODEL FROM FILE
    model = load_model(fileName)
    
    #PREPARE TEST DATA
    inputTest = np.asarray(inputTest)
    
    #TEST MODEL: with gived array and model name loaded
    yhat = model.predict(inputTest, verbose=0)
    
    print('Result per line: ',yhat.round(decimals=2))
    
    print('****End of method predictFromSavedModel\n')        


def sentimentAnalysis():
    print('****Start of method sentimentAnalysis')
    print('****End of method sentimentAnalysis\n')
    

#RESHAPER TRAINING DATA
def get_train(Input, output):
    #RESHAPE INPUT
    X = np.full((len(Input), len(Input[0]),len(Input[0][0])), 0)
    i = 0
    while i<len(Input):
        y = 0
        while y < len(Input[0]):
            X[i][y] = Input[i][y]
            y+=1
        i+=1
    
    #RESHAPE OUTPUT
    Y = np.asarray(output)
    Y = Y.reshape((len(Y), 7))
    
    return X, Y 
   
    