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

#CHOOSE THE FLAG VALUE: 1 for loading already existing model, 0 for creating a new one
flag = 0

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)
'''print(len(AllAudioNames))
print(AllAudioNames)
print(EmoCode)
print(encodedText)'''

i = 0
while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i][0])
    
    if EmoCode[i][6] != 1:
        #print('Current file:', audioFilePath)
        
        #BUILD THE OUTPUT TRAINING ARRAY: dim = (#audiofile, outlabelArray)
        TOutArray.append(EmoCode[i])
        
        #AUDIO
        #READ AUDIO FILE: tranform it in a redable array in spectrum
        arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath+'.wav')
        allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
        #BUILD THE INPUT TRAINING ARRAY: dim = (#audiofile, #ofFftPerformed, fftWindowSize)
        TInArrayAudio.append(allFrameFFT)
        
        #TEXT
        #BUILD THE INPUT TRAINING ARRAY        
        X = encodedText[i].reshape(len(encodedText[i]), 1)
        TInArrayText.append(X)
        
        #FEED THE NN: done for 1 session at time, because the groupped audio file array contains only one session files
        if flag > 0:
            modelRNNAudio = load_model(modelPath1) 
            modelRNNText = load_model(modelPath2)     
            modelAudio = nn.RNNModelAudio(modelRNNAudio, TInArrayAudio, TOutArray)  
            modelText = nn.RNNModelText(modelRNNText, TInArrayText, TOutArray)
        else:
            print('CREATE NEW MODEL FILE FOR SAVE\n')
            modelAudio = nn.RNNModelAudio('', TInArrayAudio, TOutArray)
            modelText = nn.RNNModelText('', TInArrayText, TOutArray)
            flag +=1
          
        modelAudio.save(modelPath1, overwrite=True)
        modelText.save(modelPath2, overwrite=True)    
    
    i +=1
    TInArrayText = []
    TInArrayAudio = []
    TOutArray = []
            
print('END OF TRAINING V2')  