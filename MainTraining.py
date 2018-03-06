#Classe Main. Lancia l'esecuzione di tutto il processo.

import os
from DataTrainingUtils import DataTrainingUtils
import AudioUtils as aud
import NeuralNetworkUtils as nn

#SET VARIABLES AND CLASSES
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus')
dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
trainData = DataTrainingUtils()

#CREATE TRAINING OUTPUT DATA FILE
trainData.setDataCorpus()

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
            
            #READ AUDIO FILE: tranform it in a redable array in spectrum
            x1 = aud.getArrayFromAudio(Afile)
            x2 = aud.getFrameArray(x1)
            input = aud.getSpectrumFrameArray(x2)
           
            #READ TRAINING OUTPUT DATA: corresponding to that audio file
            output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
            print('---Coresponding output for Audio Ses04F_script01_1_M019---')
            print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text) 
           
            #FEED THE NN
            y = nn.FFNNModel(input, output)
            
            
            


