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
    Y = Y.reshape(len(Y), 7)
    
    #PRINT INFO ON INPUTS AND OUTPUTS
    print('X shape', X.shape)
    print('Y shape', Y.shape)
    '''print('X: ', X)
    print('Y: ', Y)'''
    
    #DEFINE MODEL: if model do not exist create it otherwise use the given one. (#audioFile,#timestep,#fftvalues)
    if modelRNNAudio == '':
        model = Sequential()
        
        #BATCH SIZE > 1
        #model.add(LSTM(64, input_shape=(len(X[0]),len(X[0][0])), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        #BATCH SIZE 1
        model.add(LSTM(64, input_shape=(None,len(X[0][0])), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        
        model.add(Dense(7, activation='sigmoid'))#activation='softmax'
        model.compile(loss='categorical_crossentropy', optimizer='adam')#binary_crossentropy
    else:
        model = modelRNNAudio 
    
    #START MODEL    
    model.fit(X, Y, epochs=10, batch_size=len(X), show_accuracy=True)
    
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
    
    
def predictFromSavedModel(modelFilePath, inputTest):
    print('****Start of method predictFromSavedModel')
    
    #LOAD MODEL FROM FILE
    model = load_model(modelFilePath)
    
    #PREPARE TEST DATA
    inputTest = np.asarray(inputTest)
    
    #TEST MODEL: with gived array and model name loaded
    yhat = model.predict(inputTest, verbose=0)
    
    print('Result per line: ',yhat.round(decimals=3))
    
    print('****End of method predictFromSavedModel\n')   
    return yhat.round(decimals=3)     


def evaluateModel(modelFilePath, inputTest, outputTest):
    print('****Start of method evaluateModel')
    
    #PREPARE INPUT AND OUTPUT
    X = np.asarray(inputTest) 
    Y = np.asarray(outputTest)
    
    #LOAD MODEL FROM FILE
    model = load_model(modelFilePath)
    
    # Final evaluation of the model
    scores = model.evaluate(X, Y, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    print('****End of method evaluateModel\n')
    

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
   
    