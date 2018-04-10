import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.models import load_model


def reshapeLSTMInOut(audFeat, label):
    X = []
    X.append(audFeat)
    X = np.asarray(X)
    Y = np.asarray(label)    
    return X, Y


def readFeatures(DirRoot):
    listA = [ item for item in os.listdir(DirRoot) if os.path.isfile(os.path.join(DirRoot, item)) ]
    allFileFeature = []
    allFileName = []
    
    #READ encoded audio Features
    for file in listA:
        allFileName.append(file)
        datareader = csv.reader(open(os.path.join(DirRoot, file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        
        #Append all files feature in an unique array
        allFileFeature.append(Y)
        
    allFileFeature = np.asarray(allFileFeature)
    
    return allFileFeature, allFileName


def predictFromSavedModel(modelFilePath, inputTest, labels, fileName, limit):
    
    #LOAD MODEL FROM FILE
    model = load_model(modelFilePath)
    
    #FOR EACH FILE PREDICT
    for i in range(limit):
        if labels[i][0][6] != 2: 
            print('Current file:', fileName[i])
            
            #FORMAT X & Y
            X, Y = reshapeLSTMInOut(inputTest[i], labels[i])
            
            #PREDICT
            yhat = model.predict(X, verbose=0)
            #yhat = model.predict_classes(X, verbose=0)
    
            print('Expected:', Y, 'Predicted', yhat)


def predictFromSavedModelV2(modelFilePath, inputAudio, inputText, labels, fileName, limit):
    #LOAD MODEL FROM FILE
    model = load_model(modelFilePath)
    
    #FOR EACH FILE PREDICT
    for i in range(limit):
        if labels[i][0][6] != 2: 
            print('Current file:', fileName[i])
            
            #FORMAT X & Y
            X, Y = reshapeLSTMInOut(inputAudio[i], labels[i])
            
            #PREDICT
            #yhat = model_Audio.predict(X, verbose=0)
            yhat = model.predict_classes(X, verbose=0)
    
            print('Expected:', Y, 'Predicted', yhat)


if __name__ == '__main__':
    
    #DEFINE MAIN ROOT
    mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    #mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    
    #BUILD PATH FOR EACH FEATURE DIR
    dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
    dirText = os.path.join(mainRoot + '\FeaturesText')
    dirLabel = os.path.join(mainRoot + '\LablesEmotion')
    
    #BUILD MODELS PATH
    mainRootModelAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
    mainRootModelText = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
    
    #EXTRACT FEATURES AND LABELS
    allAudioFeature, allFileName = readFeatures(dirAudio)
    allTextFeature, allFileName = readFeatures(dirText)
    allLabels, allFileName = readFeatures(dirLabel)
    print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)
    
    #DEFINE BLSTM MODEL
    modelType = 0 #1=OnlyAudio, 2=OnlyText, 3=Audio&Text
    limit = 5 #number of file trained: len(allAudioFeature) or a number
    
    #TRAIN & SAVE LSTM: considering one at time
    if modelType == 0:
        predictFromSavedModel(mainRootModelAudio, allAudioFeature, allLabels, allFileName, limit)
    if modelType == 1:
        predictFromSavedModel(mainRootModelText, allTextFeature, allLabels, allFileName, limit) 
    if modelType == 2:
        predictFromSavedModelV2(mainRootModelText, allAudioFeature, allTextFeature, allLabels, allFileName, limit)         
    
    
    