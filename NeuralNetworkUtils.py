import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

    
def RNNModelAudio(modelRNNAudio, Input, output):
    print('****Start of method RNNModelAudio')
    
    #PREPARE INPUT AND OUTPUT
    X = np.asarray(Input) 
    Y = np.asarray(output)
    Y = Y.reshape((len(Y), 7))
    
    #PRINT INFO ON INPUTS AND OUTPUTS
    '''print('len X1: ', len(X))
    print('len X2: ', len(X[0]))
    print('len X3: ', len(X[0][0]))
    print('X: ', X)
    print('len Y1: ', len(Y))
    print('len Y2: ', len(Y[0]))
    print('Y: ', Y)'''
    
    #DEFINE MODEL: if model do not exist create it otherwise use the given one.
    if modelRNNAudio == '':
        model = Sequential()
        model.add(LSTM(64, input_shape=(None,len(X[0][0])), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    else:
        model = modelRNNAudio 
        
    model.fit(X, Y, epochs=10,verbose=0)
    
    print('****End of method RNNModelAudio\n')
    return model


def RNNModelText(modelRNNText, Input, output):
    print('****Start of method RNNModelText')
    
    #PREPARE INPUT AND OUTPUT
    X = np.asarray(Input) 
    Y = np.asarray(output)
    Y = Y.reshape((len(Y), 7))
    
    #PRINT INFO ON INPUTS AND OUTPUTS
    '''print('len X1: ', len(X))
    print('len X2: ', len(X[0]))
    print('X: ', X)
    print('len Y1: ', len(Y))
    print('len Y2: ', len(Y[0]))
    print('Y: ', Y)'''
    
    #DEFINE MODEL: if model do not exist create it otherwise use the given one.
    if modelRNNText == '':
        model = Sequential()
        model.add(LSTM(100, input_shape=(None,1), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    else:
        model = modelRNNText 
        
    model.fit(X, Y, epochs=10,verbose=0)
    
    print('****End of method RNNModelText\n')
    return model
    
    
def predictFromSavedModel(inputTest, fileName):
    print('****Start of method predictFromSavedModel')
    
    #LOAD MODEL FROM FILE
    model = load_model(fileName)
    
    #PREPARE TEST DATA
    inputTest = np.asarray(inputTest)
    
    #TEST MODEL: with gived array and model name loaded
    yhat = model.predict(inputTest, verbose=0)
    
    print('Result per line: ',yhat.round(decimals=3))
    
    print('****End of method predictFromSavedModel\n')   
    return yhat.round(decimals=3)     


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
   
    