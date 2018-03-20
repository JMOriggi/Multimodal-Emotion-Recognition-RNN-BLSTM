import os
import numpy as np
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET VARIABLES AND CLASSES
#mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_lav2')
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Predict')
dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
TInArrayTest = []

#CREATE TRAINING OUTPUT DATA FILE
#trainData.setDataCorpus()

#MAIN ROUTINE: load one after the other all the audio files for each session
for session in dirlist:
    currentSessionPathText = os.path.join(mainRoot, session)
    currentSessionPathText += '\Sentences_audio'
    directoryAudio = os.path.normpath(currentSessionPathText)
    #print(directoryAudio)
    for dirs, subdir, files in os.walk(directoryAudio):
        #print('Directory: ',dirs)
        for Afile in files:
            #print('Current File: ',Afile)
            audioFilePath = os.path.join(currentSessionPathText, Afile) 
            
            #READ AUDIO FILE: tranform it in a redable array in spectrum
            arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath)
            allFrameFFT = aud.getFreqArray(arrayAudio, sampleRate)
            
            #READ TRAINING OUTPUT DATA: corresponding to that audio file
            y_code, output, emo, val, text = trainData.getOutputDataFromAudio(Afile.split('.')[0])
            print('---Coresponding output for Audio ', Afile)
            print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text,'Emo Label code: ', y_code, '\n')
            
            #BUILD THE INPUT TRAINING ARRAY: dim = (#audiofile, #ofFftPerformed, fftWindowSize)
            TInArrayTest.append(allFrameFFT)
            
            #PREPARE TRAINING DATA
            def get_train():
                X = np.full((len(TInArrayTest), len(TInArrayTest[0]),len(TInArrayTest[0][0])), 0)
                
                #Reshape
                i = 0
                while i<len(TInArrayTest):
                    y = 0
                    while y < len(TInArrayTest[0]):
                        X[i][y] = TInArrayTest[i][y]
                        y+=1
                    i+=1
                
                return X
            
            #GET DATA FOR TRAINING
            X = get_train()
            
            #TEST MODEL
            nn.predictFromSavedModel(X, 'RNN_Model_saved.h5')
            
            #RESET ARRAY: se non viene fatto accumulo in tin e tout tutti i file audio (risultato finale voluto, eseguendo la RNN all'uscita di tutto il ciclo)
            TInArrayTest = []  
            

