import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn
from keras.models import load_model

#SET MAIN ROOT
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')

#SET VARIABLES AND CLASSES
audioDirectoryPath = os.path.normpath(mainRoot + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
TInArray = []
TOutArray = []

#CHOOSE THE FLAG VALUE: 1 for loading already existing model, 0 for creating a new one
flag = 0
flag = 1

#LOAD DATA FOR TRAINING
AllAudioNames, EmoCode, encodedText = trainData.readCsvData(mainRoot)

i=0
while i < len(AllAudioNames):
    audioFilePath = os.path.join(audioDirectoryPath, AllAudioNames[i])
    
    if EmoCode[i] != [0,0,0,0,0,0,1]:
        print('Current file:', audioFilePath)
        
        #READ AUDIO FILE: tranform it in a redable array in spectrum
        arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath)
        allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
        
        #BUILD THE INPUT TRAINING ARRAY: dim = (#audiofile, #ofFftPerformed, fftWindowSize)
        TInArray.append(allFrameFFT)
        
        #BUILD THE OUTPUT TRAINING ARRAY: dim = (#audiofile, outlabelArray)
        TOutArray.append(EmoCode)
        
        '''print('\n')
        print('TInArray number of audio file: ', len(TInArray))
        print('TInArray number of timestep (number of FFT window): ', len(TInArray[0]))
        print('TInArray number of freq considered (value is the amplitude in db for each one): ', len(TInArray[0][0]))
        print('TOutArray number of audio file: ', len(TOutArray))
        print('TOutArray number of output label for each file: ', len(TOutArray[0]))
        print('\n')'''
        
        #FEED THE NN: done for 1 session at time, because the groupped audio file array contains only one session files
        '''if flag > 0:
            modelRNN = load_model('RNN_Model_saved.h5')    
            model = nn.RNNModel(modelRNN, TInArray, TOutArray)
        else:
            print('CREATE NEW MODEL FILE FOR SAVE\n')
            modelRNN = ''
            model = nn.RNNModel(modelRNN, TInArray, TOutArray)
            flag +=1
            
        #SAVE MODEL AND WEIGHTS AFTER TRAINING
        model.save('RNN_Model_saved.h5', overwrite=True)'''
        
        #RESET ARRAY: se non viene fatto accumulo in tin e tout tutti i file audio (risultato finale voluto, eseguendo la RNN all'uscita di tutto il ciclo)
        TInArray = []    
        TOutArray = []
        
    i +=1
            
print('END OF TRAINING V2')  