import os
import numpy as np
import DataTrainingUtils as trainData
import NeuralNetworkUtils as nn

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')

#SET PATH AND VARIABLES
modelPath = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
audioDirectoryPath = os.path.normpath(mainRoot + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
TInArrayAudio = []
TInArrayText = []
TOutArray = []

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)

#CREATE OUTPUT DATA FILE: remove if it already exist and recreate it new
resultFilePath = os.path.join(mainRoot+'\ResultsPredictionText.txt')
try:
    os.remove(resultFilePath)
except OSError:
    pass
resultFile = open(resultFilePath, 'a')

i = 0
while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i][0])
    
    #TEXT
    #BUILD THE INPUT TRAINING ARRAY        
    X = encodedText[i].reshape(len(encodedText[i]), 1)
    TInArrayText.append(X)
    
    #TEST MODEL
    textRes = nn.predictFromSavedModel(modelPath, TInArrayText) 
    
    #APPEND IN THE OUTPUT FILE
    resultLine = AllAudioNames[i][0]+',EXP:'+str(EmoCode[i])+',TEXT:'+str(textRes[0])+'\n'                    
    resultFile.writelines(resultLine)
    print(resultLine)
    
    TInArrayText = []
    i +=1

resultFile.close()    
            
print('END OF PREDICTION V1.2: Text')  

