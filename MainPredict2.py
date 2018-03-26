import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn
from keras.models import load_model

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')

#SET PATH AND VARIABLES
modelPath1 = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
modelPath2 = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')
audioDirectoryPath = os.path.normpath(mainRoot + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
TInArrayAudio = []
TInArrayText = []
TOutArray = []

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)

#CREATE OUTPUT DATA FILE: remove if it already exist and recreate it new
resultFilePath = os.path.join(mainRoot+'\Results.txt')
try:
    os.remove(resultFilePath)
except OSError:
    pass
resultFile = open(resultFilePath, 'a')

i = 0
while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i][0])
    
    #if EmoCode[i][6] != 1: #Don't consider neu,xx and other emotion label
    #READ AUDIO FILE: tranform it in a redable array in spectrum
    arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath+'.wav')
    allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
    TInArrayAudio.append(allFrameFFT)
    
    #TEXT
    #BUILD THE INPUT TRAINING ARRAY        
    X = encodedText[i].reshape(len(encodedText[i]), 1)
    TInArrayText.append(X)
    
    #TEST MODEL
    audioRes = nn.predictFromSavedModel(TInArrayAudio, modelPath1)
    textRes = nn.predictFromSavedModel(TInArrayText, modelPath2) 
    print('Expected result: ',EmoCode[i])     
    print(audioRes[0])
    print(textRes[0])
    
    #APPEND IN THE OUTPUT FILE
    resultLine = AllAudioNames[i][0]+',EXP:'+str(EmoCode[i])+',AUD:'+str(audioRes[0])+',TEXT:'+str(textRes[0])+'\n'                    
    resultFile.writelines(resultLine)
    
    TInArrayText = []
    TInArrayAudio = []
    i +=1

resultFile.close()    
            
print('END OF PREDICTION V2')  

