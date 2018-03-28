import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn
from keras.models import load_model

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
#mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Training')

#SET PATH AND VARIABLES
modelPath = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
audioDirectoryPath = os.path.normpath(mainRoot + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
TInArrayAudio = []
TOutArray = []
i = 0

#CHOOSE THE FLAG VALUE (for single batch mode): 1 for loading already existing model, 0 for creating a new one
flag = 0

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)
'''print('Len AllAudioNames: ',len(AllAudioNames))
print('Len EmoCode: ',len(EmoCode))
print('Len encodedText: ',len(encodedText))
print('AllAudioNames: ',AllAudioNames)
print('EmoCode: ',EmoCode)
print('encodedText: ',encodedText)'''

while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i][0])

    #DON'T EVALUATE labels 'xxx' and 'other': code[0,0,0,0,0,0,2]  
    if EmoCode[i][6] != 2: 
        print('Current file:', audioFilePath)
        
        #BUILD THE OUTPUT TRAINING ARRAY: dim = (#audiofile, outlabelArray)
        TOutArray.append(EmoCode[i])
        
        #AUDIO: Read audio file and tranform it in dim = (#audiofile, #ofFftPerformed, fftWindowSize)
        arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath+'.wav')
        #allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate) #BATCH SIZE > 1 MODE
        allFrameFFT = aud.getFreqArrayV2(arrayAudio, sampleRate) #BATCH SIZE 1 MODE
        TInArrayAudio.append(allFrameFFT)
        
        #SINGLE BATCH TRAINING: if flag=0 at the first iteration it creates the model, otherwise load an existing model
        if flag > 0:
            modelRNNAudio = load_model(modelPath)     
            modelAudio = nn.RNNModelAudio(modelRNNAudio, TInArrayAudio, TOutArray)  
        else:
            print('CREATE NEW MODEL FILE FOR SAVE\n')
            modelAudio = nn.RNNModelAudio('', TInArrayAudio, TOutArray)
            flag +=1
        modelAudio.save(modelPath, overwrite=True) 
        TInArrayAudio = []
        TOutArray = []
        
    i +=1

#CHECK LISTS SHAPE
'''print('TInArrayAudio shape: ',np.asarray(TInArrayAudio).shape)
print('TOutArray shape: ',np.asarray(TOutArray).shape)'''

#ALL BATCH TRAINING
'''print('CREATE NEW MODEL FILE FOR SAVE\n')
modelAudio = nn.RNNModelAudio('', TInArrayAudio, TOutArray)
modelAudio.save(modelPath, overwrite=True)'''
            
print('END OF TRAINING V1.1: Audio')  



