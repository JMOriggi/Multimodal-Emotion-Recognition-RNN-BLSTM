import numpy as np
import csv
from numpy import genfromtxt
import os
import librosa
import operator
from scipy.io.wavfile import read
import librosa
import matplotlib.pyplot as plt 
import calculate_features as c_f


def readWav(audioFilePath):
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    monoAudio = inputAudio[1]
    monoAudio = np.array(monoAudio, dtype=float) #type float necessary
    return monoAudio, sampleRate

main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Training')

all_wav_path = os.path.join(main_root + '\AllAudioFiles')
audioDirectoryPath = os.path.normpath(main_root + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
print(audlist[30])

#CURRENT FILE FEATURE
audioFilePath = os.path.join(audioDirectoryPath, audlist[30])
arrayAudio, sampleRate = readWav(audioFilePath)
print(arrayAudio)
print(sampleRate)
currentFileFeatures = c_f.calculate_features(arrayAudio, sampleRate, False).T

print(currentFileFeatures)
print(currentFileFeatures.shape)



audioFilePath = os.path.join(audioDirectoryPath, audlist[31])
arrayAudio, sampleRate = readWav(audioFilePath)
print(arrayAudio)
print(sampleRate)
currentFileFeatures = c_f.calculate_features(arrayAudio, sampleRate, False).T

print(currentFileFeatures)
print(currentFileFeatures.shape)






