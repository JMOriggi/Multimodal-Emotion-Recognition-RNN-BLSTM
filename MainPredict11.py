import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')

#SET PATH AND VARIABLES
modelPath = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
audioDirectoryPath = os.path.normpath(mainRoot + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
TInArrayAudio = []
TOutArray = []

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)

#CREATE OUTPUT DATA FILE: remove if it already exist and recreate it new
resultFilePath = os.path.join(mainRoot+'\ResultsPredictionAudio.txt')
try:
    os.remove(resultFilePath)
except OSError:
    pass
resultFile = open(resultFilePath, 'a')

i = 0
while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i][0])
    
    if EmoCode[i][6] != 2:
        #READ AUDIO FILE: tranform it in a redable array in spectrum
        arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath+'.wav')
        #allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate) #BATCH SIZE > 1 MODE
        allFrameFFT = aud.getFreqArrayV2(arrayAudio, sampleRate) #BATCH SIZE 1 MODE
        TInArrayAudio.append(allFrameFFT)
        
        #TEST MODEL
        audioRes = nn.predictFromSavedModel(modelPath, TInArrayAudio)
        
        #APPEND IN THE OUTPUT FILE
        resultLine = AllAudioNames[i][0]+',EXP:'+str(EmoCode[i])+',AUD:'+str(audioRes[0])+'\n'                    
        resultFile.writelines(resultLine)
        print(resultLine)
        
        TInArrayAudio = []
    i +=1

resultFile.close()    
            
print('END OF PREDICTION V1.1: Audio')  

