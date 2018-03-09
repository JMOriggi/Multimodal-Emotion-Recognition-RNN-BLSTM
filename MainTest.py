#TEST DATA TRAINING CLASS
#from DataTrainingUtils import DataTrainingUtils
#trainData = DataTrainingUtils()
#trainData.setDataCorpus()
#output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
#print('---Coresponding output for Audio Ses04F_script01_1_M019---')
#print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text)            
 
#TEST AUDIO CLASS     
import AudioUtils as aud
import NeuralNetworkUtils as nn
import numpy as np

audioFileName ='file.wav'
arrayAudio, sampleRate = aud.getArrayFromAudio(audioFileName)
allFrame = aud.getFrameArray(arrayAudio, sampleRate, 1024)
allFrameFFT = aud.getSpectrumFrameArray(allFrame)

#nn.FFNNModel(np.float32(allFrameFFT[0]), 1)
