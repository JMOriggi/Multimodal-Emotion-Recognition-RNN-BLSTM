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
i = 0

#CHOOSE THE FLAG VALUE: 1 for loading already existing model, 0 for creating a new one
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
        '''print('Current file:', audioFilePath)'''
        
        #BUILD THE OUTPUT TRAINING ARRAY: dim = (#audiofile, outlabelArray)
        TOutArray.append(EmoCode[i].tolist())
        
        #AUDIO
        #Read audio file: tranform it in a redable list in spectrum
        #Build input list: dim = (#audiofile, #ofFftPerformed, fftWindowSize)
        arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath+'.wav')
        allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
        TInArrayAudio.append(allFrameFFT.tolist())#very important to have a list casting!!!
        
        #TEXT
        #Build input array        
        X = encodedText[i].reshape(len(encodedText[i]), 1)
        TInArrayText.append(X.tolist())
        
        #FEED THE NN: if flag=0 at the first iteration it creates the model, otherwise load an existing model
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
        
        #SAVE THE 2 MODEL TRAINED  
        modelAudio.save(modelPath1, overwrite=True)
        modelText.save(modelPath2, overwrite=True)
        
        #RESET ARRAYS AND INCREMENT i
        TInArrayText = []
        TInArrayAudio = []
        TOutArray = []
        
    i +=1          

print('END OF TRAINING V2: Audio + Text')  


