import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn
from keras.models import load_model

#SET VARIABLES AND CLASSES
mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
#mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
sessDirList = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
TInArray = []
TOutArray = []
flag = 0

#MAIN ROUTINE: load one after the other all the audio files for each session
for session in sessDirList:
    currentAudioDirPath = os.path.normpath(os.path.join(mainRoot, session)+'\Sentences_audio')
    audioGroupDir = [ item for item in os.listdir(currentAudioDirPath) if os.path.isdir(os.path.join(currentAudioDirPath, item)) ]
    
    for audioGroup in audioGroupDir:
        currentAudioGroupPath = os.path.normpath(os.path.join(currentAudioDirPath, audioGroup))
        audlist = [ item for item in os.listdir(currentAudioGroupPath) if os.path.isfile(os.path.join(currentAudioGroupPath, item)) ]
            
        for Afile in audlist:
            audioFilePath = os.path.join(currentAudioGroupPath, Afile) 
            
            #READ TRAINING OUTPUT DATA: corresponding to the current audio file; if emotion xxx, neu or other don't consider it for training
            y_code, output, emo, val, text = trainData.getOutputDataFromAudio(Afile.split('.')[0], mainRoot)
            
            if y_code != [0,0,0,0,0,0,1]:
                #READ AUDIO FILE: tranform it in a redable array in spectrum
                arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath)
                allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
                
                #BUILD THE INPUT TRAINING ARRAY: dim = (#audiofile, #ofFftPerformed, fftWindowSize)
                TInArray.append(allFrameFFT)
                
                #BUILD THE OUTPUT TRAINING ARRAY: dim = (#audiofile, outlabelArray)
                TOutArray.append(y_code)
                
                '''print('\n')
                print('TInArray number of audio file: ', len(TInArray))
                print('TInArray number of timestep (number of FFT window): ', len(TInArray[0]))
                print('TInArray number of freq considered (value is the amplitude in db for each one): ', len(TInArray[0][0]))
                print('TOutArray number of audio file: ', len(TOutArray))
                print('TOutArray number of output label for each file: ', len(TOutArray[0]))
                print('\n')'''
                
                #FEED THE NN: done for 1 session at time, because the groupped audio file array contains only one session files
                if flag > 0:
                    modelRNN = load_model('RNN_Model_saved.h5')    
                    model = nn.RNNModel(modelRNN, TInArray, TOutArray)
                else:
                    print('CREATE NEW MODEL FILE FOR SAVE\n')
                    modelRNN = ''
                    model = nn.RNNModel(modelRNN, TInArray, TOutArray)
                    flag +=1
                    
                #SAVE MODEL AND WEIGHTS AFTER TRAINING
                model.save('RNN_Model_saved.h5', overwrite=True)
                
                #RESET ARRAY: se non viene fatto accumulo in tin e tout tutti i file audio (risultato finale voluto, eseguendo la RNN all'uscita di tutto il ciclo)
                TInArray = []    
                TOutArray = []
            
print('END OF TRAINING')  

          