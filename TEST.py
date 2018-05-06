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

main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Training')

all_wav_path = os.path.join(main_root + '\AllAudioFiles')
audioDirectoryPath = os.path.normpath(main_root + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
print(audlist[30])

'''#CURRENT FILE FEATURE
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
print(currentFileFeatures.shape)'''

#CURRENT FILE FEATURE
audioFilePath = os.path.join(audioDirectoryPath, audlist[30])
arrayAudio, sampleRate = readWav(audioFilePath)
print(arrayAudio)
print(sampleRate)
window = int(0.020*sampleRate)
hop = int(0.010*sampleRate)

#COMPUTE SPECTROGRAM: NFFT=how many sample in one chunk, for Fs16000 chunks20ms-->32
fft, freqsBins, timeBins, im = plt.specgram(arrayAudio, Fs=sampleRate, NFFT=window, noverlap=hop, cmap=plt.get_cmap('autumn_r'))

#PRINT INFO
print('shape fft: ', fft.shape)
print(fft)
fft = fft.T
fft = fft[:,6:61] #Select from 300-3000 HZ - len(55bins)
fft = fft.T
print('shape fft: ', fft.shape)
print(fft)
freqsBins = freqsBins[6:61] #Select from 300-3000 HZ - len(55bins)
print('shape fft: ', fft.shape)
print('shape timeBins ', timeBins.shape)
print('shape freqsBins: ', freqsBins.shape)
print('freqsBins: ', freqsBins)





