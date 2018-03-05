#Classe Main. Lancia l'esecuzione di tutto il processo.

import os
from DataTrainingUtils import DataTrainingUtils
from AudioUtils import AudioUtils
from NeuralNetworkUtils import NeuralNetworkUtils


mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus')
dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
trainData = DataTrainingUtils()
audioUtils = AudioUtils()
NN = NeuralNetworkUtils()


#trainData.setDataCorpus()

for session in dirlist:
    currentSessionPathText = os.path.join(mainRoot, session)
    currentSessionPathText += '\Sentences_audio'
    directoryAudio = os.path.normpath(currentSessionPathText)
    print(directoryAudio)
    for dirs, subdir, files in os.walk(directoryAudio):
        #print('Directory: ',dirs)
        for Afile in files:
            print('Current File: ',Afile)
            #getSpetrumFromAudio
            #getnextFrame
            #NNModel
            #x= NN output
            #y=getOutputDataFromAudio
            #LossError(x-y)
            
#output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
#print('---Coresponding output for Audio Ses04F_script01_1_M019---')
#print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text) 


