import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional



if __name__ == '__main__':
    
    #DEFINE MAIN ROOT
    mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    #mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    
    #BUILD PATH FOR EACH FEATURE DIR
    dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
    dirText = os.path.join(mainRoot + '\FeaturesText')
    dirLabel = os.path.join(mainRoot + '\LablesEmotion')
    
    #EXTRACT FEATURES AND LABELS
    '''allAudioFeature, allFileName = readFeatures(dirAudio)
    allTextFeature, allFileName = readFeatures(dirText)
    allLabels, allFileName = readFeatures(dirLabel)
    print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)
    
    #DEFINE BLSTM MODEL
    model = buildBLTSM()
    modelType = 0 #1=OnlyAudio, 2=OnlyText, 3=Audio&Text
    limit = 5 #number of file trained: len(allAudioFeature) or a number
    n_epoch = 10 #number of epoch for each file trained
    
    #TRAIN & SAVE LSTM: considering one at time
    if modelType == 0 or modelType == 2:
        model_Audio = trainBLSTM(allFileName, allAudioFeature, allLabels, model, limit, n_epoch)
        modelPathAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
        model_Audio.save(modelPathAudio, overwrite=True)
    if modelType == 1 or modelType == 2:
        modelText = trainBLSTM(allFileName, allTextFeature, allLabels, model, limit, n_epoch)   
        modelPathAudio = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
        model_Audio.save(modelPathAudio, overwrite=True)    
    
    #EVALUATE LSTM
    X, Y = reshapeLSTMInOut(allAudioFeature[2], allLabels[2])
    #yhat = model_Audio.predict(X, verbose=0)
    yhat = model_Audio.predict_classes(X, verbose=0)
    print('Expected:', Y, 'Predicted', yhat)'''
    