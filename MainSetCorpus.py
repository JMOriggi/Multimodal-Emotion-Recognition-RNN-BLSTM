import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET ROOTS
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRootTraining = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
#mainRootTest = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Test')

#CREATE TRAINING OUTPUT DATA FILE
trainData.setDataCorpus(mainRootTraining)

#CREATE TEST OUTPUT DATA FILE
#trainData.setDataCorpus(mainRootTest)

print('END OF SET CORPUS')
