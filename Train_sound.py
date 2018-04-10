import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional


def readFeatures(DirRoot):
    listA = [ item for item in os.listdir(DirRoot) if os.path.isfile(os.path.join(DirRoot, item)) ]
    allFileFeature = []
    
    #READ encoded audio Features
    for file in listA:
        datareader = csv.reader(open(os.path.join(DirRoot, file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        
        #Append all files feature in an unique array
        allFileFeature.append(Y)
        
    allFileFeature = np.asarray(allFileFeature)
    
    return allFileFeature 


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
        

def trainBLSTM(Features, Labels, model, limit):
    
    for i in range(limit):
        #Format correctly single input and output
        X, Y = reshapeLSTMInOut(Features[i], Labels[i])
        
        #FIT MODEL for one epoch on this sequence
        model.fit(X, Y, epochs=20, batch_size=1, verbose=2)
    
    return model  

def trainBLSTMV2(Features, Labels, model, limit):
    
    for i in range(limit):
        #Format correctly single input and output
        X, Y = reshapeLSTMInOut(Features[i], Labels[i])
        
        #FIT MODEL for one epoch on this sequence
        model.fit(X, Y, epochs=20, batch_size=1, verbose=2)
    
    return model     

    
if __name__ == '__main__':
    
    mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    #mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    
    #BUILD PATH FOR EACH FEATURE DIR
    dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
    dirText = os.path.join(mainRoot + '\FeaturesText')
    dirLabel = os.path.join(mainRoot + '\LablesEmotion')
    
    #EXTRACT FEATURES AND LABELS
    allAudioFeature = readFeatures(dirAudio)
    allTextFeature = readFeatures(dirText)
    allLabels = readFeatures(dirLabel)
    print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)
    
    #DEFINE BLSTM MODEL Parameters
    modelType = 0 #1=OnlyAudio, 2=OnlyText, 3=Audio&Text
    limit = 5 #number of file trained: len(allAudioFeature) or a number
    
    modelA = buildBLTSM()
    modelT = buildBLTSM()
    
    #TRAIN LSTM: considering one at time
    if modelType == 0:
        modelA = trainBLSTM(allAudioFeature, allLabels, modelA, limit)
    if modelType == 1:
        modelT = trainBLSTM(allTextFeature, allLabels, modelT, limit)       
    
    #SAVE MODEL
    modelPathAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
    modelA.save(modelPathAudio, overwrite=True)
    
    #EVALUATE LSTM
    X, Y = reshapeLSTMInOut(allAudioFeature[2], allLabels[2])
    #yhat = model.predict(X, verbose=0)
    yhat = modelA.predict_classes(X, verbose=0)
    print('Expected:', Y, 'Predicted', yhat)
    
    