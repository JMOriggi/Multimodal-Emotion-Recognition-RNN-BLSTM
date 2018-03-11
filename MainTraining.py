#Classe Main. Lancia l'esecuzione di tutto il processo.

import os
import DataTrainingUtils as trainData
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET VARIABLES AND CLASSES
mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus')
#mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus')
dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]

#CREATE TRAINING OUTPUT DATA FILE
#trainData.setDataCorpus()

#MAIN ROUTINE: load one after the other all the audio files for each session
#for session in dirlist[0]:
for session in dirlist:
    currentSessionPathText = os.path.join(mainRoot, session)
    currentSessionPathText += '\Sentences_audio'
    directoryAudio = os.path.normpath(currentSessionPathText)
    print(directoryAudio)
    for dirs, subdir, files in os.walk(directoryAudio):
        #print('Directory: ',dirs)
        for Afile in files:
            print('Current File: ',Afile)
            audioFilePath = os.path.join(currentSessionPathText, Afile) 
            
            #READ AUDIO FILE: tranform it in a redable array in spectrum
            arrayAudio, sampleRate = aud.getArrayFromAudio(audioFilePath)
            allFrame = aud.getFrameArray(arrayAudio, sampleRate, 1024)
            print('allFrame: ', allFrame)
            print('allFrame type: ', type(allFrame))
            allFrameFFT = aud.getSpectrumFrameArray(allFrame)
            print('allFrameFFT: ', allFrameFFT)
            print('allFrameFFT type: ', type(allFrameFFT))
           
            #READ TRAINING OUTPUT DATA: corresponding to that audio file
            #y_code, output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
            #print('---Coresponding output for Audio Ses04F_script01_1_M019---')
            y_code, output, emo, val, text = trainData.getOutputDataFromAudio(Afile.split('.')[0])
            print('---Coresponding output for Audio ', Afile)
            print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text,'Emo Label code: ', y_code)
           
            #FEED THE NN
            #y = nn.FFNNModel(allFrameFFT, y_code)
            nn.RNNModel(allFrameFFT, y_code)
            
            



