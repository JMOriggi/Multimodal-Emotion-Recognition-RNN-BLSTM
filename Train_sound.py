import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional


def readFeature():
    mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus\FeaturesAudio')
    listA = [ item for item in os.listdir(mainRoot) if os.path.isfile(os.path.join(mainRoot, item)) ]
    allFileFeature = []
    
    #READ encoded audio Features
    for file in listA:
        datareader = csv.reader(open(os.path.join(mainRoot,file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        
        #Append all files feature in an unique array
        allFileFeature.append(Y)
        
    allFileFeature = np.asarray(allFileFeature)
    
    return allFileFeature 


def readLabels():
    mainRoot2 = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus\LablesEmotion')
    listE = [ item for item in os.listdir(mainRoot2) if os.path.isfile(os.path.join(mainRoot2, item)) ]
    allLabels = []
    
    #READ encoded emotion: Read the content as an array of numbers and not string as default
    for file in listE:
        datareader = csv.reader(open(os.path.join(mainRoot2,file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        allLabels.append(Y)
        #print(Y.shape)
        
    allLabels = np.asarray(allLabels)
    
    return allLabels


def getListshape(X):
    X = np.asarray(X)
    return X.shape   
    

def reshapeLSTMInOut(audFeat, label):
    X = []
    X.append(audFeat)
    X = np.asarray(X)
    Y = np.asarray(label)    
    return X, Y


def buildBLTSM():
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=False), input_shape=(None, 129)))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model
        
    
if __name__ == '__main__':
    
    #EXTRACT FEATURES AND LABELS
    allFileFeature = readFeature()
    allLabels = readLabels()
    print(allFileFeature.shape)
    print(allLabels.shape)
    
    #DEFINE LSTM MODEL
    model = buildBLTSM()
    
    #TRAIN LSTM: considering totFile one at time
    totFile = 10
    for i in range(totFile):
        #Format correctly single input and output
        X, Y = reshapeLSTMInOut(allFileFeature[i], allLabels[i])
        
        #FIT MODEL for one epoch on this sequence
        model.fit(X, Y, epochs=20, batch_size=1, verbose=2)
        
        #counter += 1
    
        
    #EVALUATE LSTM
    X, Y = reshapeLSTMInOut(allFileFeature[2], allLabels[2])
    yhat = model.predict(X, verbose=0)
    #yhat = model.predict_classes(X, verbose=0)
    print('Expected:', Y, 'Predicted', yhat)
    
    