import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET MAIN ROOT
#mainRootTraining = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
#mainRootTraining = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Training')
#mainRootTest = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test')


def mainSetCorpus(mainRootTraining, mainRootTest):
    
    #TRAINING FOLDER
    #Move copy of audio files
    trainData.moveCopyAudioFiles(mainRootTraining)
    #Create training data file
    trainData.clusterData(mainRootTraining)
    trainData.encoder(mainRootTraining)
    
    #TEST FOLDER
    #Move copy of audio files
    trainData.moveCopyAudioFiles(mainRootTest)
    #Create training data file
    trainData.clusterData(mainRootTest)
    trainData.encoder(mainRootTest)
    
    print('END OF SET CORPUS')
