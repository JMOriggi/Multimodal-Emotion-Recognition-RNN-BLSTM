import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRootTraining = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
mainRootTest = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Test')

#TRAINING
#MOVE AUDIO FILES
#trainData.moveCopyAudioFiles(mainRootTraining)
#CREATE TRAINING DATA FILE
trainData.clusterData(mainRootTraining)
trainData.encoder(mainRootTraining)

'''#TEST
#MOVE AUDIO FILES
trainData.moveCopyAudioFiles(mainRootTest)
#CREATE TRAINING DATA FILE
trainData.clusterData(mainRootTest)
trainData.encoder(mainRootTest)'''

'''#CREATE TRAINING OUTPUT DATA FILE
trainData.setDataCorpus(mainRootTraining)
trainData.setDataCorpus(mainRootTest)'''

print('END OF SET CORPUS')
