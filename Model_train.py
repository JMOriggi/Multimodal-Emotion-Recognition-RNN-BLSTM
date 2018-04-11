import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.models import load_model


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
    model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(None, 129)))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model


def addEmoCount(emoLabel, counter):
    
    if  emoLabel[0][0] == 1: 
        counter[0] += 1
    if  emoLabel[0][1] == 1:    
        counter[1] += 1
    if  emoLabel[0][2] == 1: 
        counter[2] += 1
    if  emoLabel[0][3] == 1: 
        counter[3] += 1 
    if  emoLabel[0][4] == 1: 
        counter[4] += 1 
    if  emoLabel[0][5] == 1: 
        counter[5] += 1  
    if  emoLabel[0][6] == 1: 
        counter[6] += 1    
    
    return counter


def checkEmoCounter(emoLabel, counter, labelLimit):
    
    if  emoLabel[0][0] == 1: 
        if counter[0] > labelLimit:
            return 'stop'
    if  emoLabel[0][1] == 1:    
        if counter[1] > labelLimit:
            return 'stop'
    if  emoLabel[0][2] == 1: 
        if counter[2] > labelLimit:
            return 'stop'
    if  emoLabel[0][3] == 1: 
        if counter[3] > labelLimit:
            return 'stop' 
    if  emoLabel[0][4] == 1: 
        if counter[4] > labelLimit:
            return 'stop' 
    if  emoLabel[0][5] == 1: 
        if counter[5] > 0:
            return 'stop'  
    if  emoLabel[0][6] == 1: 
        if counter[6] > labelLimit:
            return 'stop'    
    
    return 'ok'

    

def trainBLSTM(fileName, Features, Labels, model, fileLimit, labelLimit, n_epoch):
    
    emoCounter = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
    
    for i in range(fileLimit):
        print('ROUND: ',i,'/',fileLimit)
        
        if Labels[i][0][6] != 2:
            print('Current file:', fileName[i])
            
            #Check number of current label processed and stop if # too high
            emoTreshStop = checkEmoCounter(Labels[i], emoCounter, labelLimit)
            print(emoTreshStop)
            
            if emoTreshStop == 'ok':
                print('train -----')
                
                #Format correctly single input and output
                X, Y = reshapeLSTMInOut(Features[i], Labels[i])
                
                #FIT MODEL for one epoch on this sequence
                model.fit(X, Y, epochs=n_epoch, batch_size=2, verbose=0)
    
                emoCounter = addEmoCount(Labels[i], emoCounter)
                print(emoCounter)
    
    return model     

    
if __name__ == '__main__':
    
    #DEFINE MAIN ROOT
    #mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    #mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    #mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
    mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Usefull')
    
    #BUILD PATH FOR EACH FEATURE DIR
    dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
    dirText = os.path.join(mainRoot + '\FeaturesText')
    dirLabel = os.path.join(mainRoot + '\LablesEmotion')
    
    #SET MODELS PATH
    mainRootModelAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
    mainRootModelText = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
    
    #EXTRACT FEATURES AND LABELS
    allAudioFeature, allFileName = readFeatures(dirAudio)
    allTextFeature, allFileName = readFeatures(dirText)
    allLabels, allFileName = readFeatures(dirLabel)
    print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)
    
    #DEFINE PARAMETERS
    modelType = 0 #1=OnlyAudio, 2=OnlyText, 3=Audio&Text
    flagLoadModel = 0 #1=load, 0=new
    fileLimit = len(allAudioFeature) #number of file trained: len(allAudioFeature) or a number
    labelLimit = 5 #Number of each emotion label file to process
    n_epoch = 15 #number of epoch for each file trained
    
    #DEFINE MODEL
    if flagLoadModel == 0:
        modelA = buildBLTSM()
        modelT = buildBLTSM()
    else:
        modelA = load_model(mainRootModelAudio)
        modelT = load_model(mainRootModelText)
    
    print('Train of #file: ', fileLimit)
    print('Train number of each emotion: ', labelLimit)
    
    #TRAIN & SAVE LSTM: considering one at time
    if modelType == 0 or modelType == 2:
        model_Audio = trainBLSTM(allFileName, allAudioFeature, allLabels, modelA, fileLimit, labelLimit, n_epoch)
        modelPathAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
        model_Audio.save(modelPathAudio, overwrite=True)       
    if modelType == 1 or modelType == 2:
        modelText = trainBLSTM(allFileName, allTextFeature, allLabels, modelT, fileLimit, labelLimit, n_epoch)    
        modelPathAudio = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
        model_Audio.save(modelPathAudio, overwrite=True)    
    
    print('END')
    
    