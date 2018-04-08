import os
import csv
import shutil
import numpy as np
import scipy
from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt 
import numpy as np 
import librosa
import pyacoustics
from pyacoustics import intensity_and_pitch
from praatio import tgio
from praatio import praatio_scripts
from praatio import pitch_and_intensity
from Signal_Analysis import features
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
from time import time
import parabolic

def estimate_pitch(monoAudio, sampleRate, window, hop):
    
    pitch , mag = librosa.core.piptrack(monoAudio, sampleRate, n_fft=window, hop_length=hop, fmin=50, fmax=1500)
    pitch = pitch.T
    mag = mag.T
    print(len(pitch))
    f0 = []
    i = 0
    while i < len(pitch):
        index = mag[i, :].argmax()
        f0.append(pitch[i, index])
        i += 1
    f0 = np.asarray(f0)
    f0 = f0.reshape((len(f0),1))
    
    return f0

def readWav(audioFilePath):
    
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    monoAudio = inputAudio[1]
    monoAudio = np.array(monoAudio, dtype=float) #type float necessary
    
    return monoAudio, sampleRate

def buildFeatures(monoAudio, sampleRate):
    #TODO: togliere primo e ultimo frame per tutte le feature
    
    window = int(0.020*sampleRate)
    hop = int(0.010*sampleRate)
    
    #PITCH:
    pitch = estimate_pitch(monoAudio, sampleRate, window, hop)
    print(pitch)
    #pitch_delta = librosa.feature.delta(pitch, width=3)
    #pitch_delta_delta = librosa.feature.delta(pitch, width=3, order=2)
    
    '''pitch = pitch.T
    mag = mag.T
    print(pitch.shape)
    print(mag.shape)
    print(mag[0])
    print(pitch[0])'''
    
    #ENERGY
    energy = librosa.feature.rmse(y=monoAudio, frame_length=window, hop_length=hop)
    energy_delta = librosa.feature.delta(energy, width=3)
    energy_delta_delta = librosa.feature.delta(energy, width=3, order=2)
    
    #MFCC
    #n_fft is the window of 25ms = 400samples e hop_lenght is the step between one frame and another
    #For the deltas important to set width=3 to compute derivative of each sample with previous and next sample.
    mfcc = librosa.feature.mfcc(monoAudio, sampleRate, n_mfcc=40, n_fft=window, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
    
    #MFCC ENERGY
    mfcc_energy = librosa.feature.rmse(S=mfcc, frame_length=window, hop_length=hop)
    mfcc_energy_delta = librosa.feature.rmse(S=mfcc_delta, frame_length=window, hop_length=hop)
    mfcc_energy_delta_delta = librosa.feature.rmse(S=mfcc_delta_delta, frame_length=window, hop_length=hop)
    
    #Transpose because librosa gives me the freq in the row but i want them in the collums
    energy = energy.T
    energy_delta = energy_delta.T
    energy_delta_delta = energy_delta_delta.T
    mfcc = mfcc.T
    mfcc_delta = mfcc_delta.T
    mfcc_delta_delta = mfcc_delta_delta.T
    mfcc_energy = mfcc_energy.T
    mfcc_energy_delta = mfcc_energy_delta.T
    mfcc_energy_delta_delta = mfcc_energy_delta_delta.T
    '''pitch = pitch.T
    pitch_delta = pitch_delta.T
    pitch_delta_delta = pitch_delta_delta.T'''
    '''print('mfcc.shape: ', mfcc.shape)
    print('mfcc_delta.shape: ', mfcc_delta.shape)
    print('mfcc_delta_delta.shape: ', mfcc_delta_delta.shape)
    print('energy.shape: ', energy.shape)
    print('energy_delta.shape: ', energy_delta.shape)
    print('energy_delta_delta.shape: ', energy_delta_delta.shape)
    print('mfcc_energy.shape: ', mfcc_energy.shape)
    print('mfcc_energy_delta.shape: ', mfcc_energy_delta.shape)
    print('mfcc_energy_delta_delta.shape: ', mfcc_energy_delta_delta.shape)'''
    '''print('pitch.shape: ', pitch.shape)
    print('pitch_delta.shape: ', pitch_delta.shape)
    print('pitch_delta_delta.shape: ', pitch_delta_delta.shape)'''
    
    
    
    #CONCATENATE FEATURES PER ROWS
    X = np.hstack((energy, energy_delta, energy_delta_delta, mfcc, mfcc_delta, mfcc_delta_delta, mfcc_energy, mfcc_energy_delta, mfcc_energy_delta_delta))
    print(X.shape)
    
    return X

def saveFeaturecsv(currentFileFeatures, csvOutputFilePath):
    csvOutputFilePath = os.path.join(csvOutputFilePath + '.csv')
    print(csvOutputFilePath)
    try:
        os.remove(csvOutputFilePath)
    except OSError:
        pass
    
    with open(csvOutputFilePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.asarray(currentFileFeatures))
    f.close()


def buildAudioCsv(mainRoot, audioDirectoryPath, out_audio_feature_path):
    currentFileFeatures = []
    audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
    for audioFile in audlist:
        audioFilePath = os.path.join(audioDirectoryPath, audioFile)
        csvOutputFilePath = os.path.join(out_audio_feature_path, audioFile.split('.')[0])
        print('Current file: ', csvOutputFilePath)
        
        arrayAudio, sampleRate = readWav(audioFilePath)
        
        currentFileFeatures = buildFeatures(arrayAudio, sampleRate)
        
        saveFeaturecsv(currentFileFeatures, csvOutputFilePath)
        
        currentFileFeatures = []
        
    
    
if __name__ == '__main__':
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    all_wav_path = os.path.join(main_root + '\AllAudioFiles')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_audio_feature_path = os.path.join(main_root+'\FeaturesAudio')   
    
    buildAudioCsv(index_file_path, all_wav_path, out_audio_feature_path)  
        
    print('****END')
        
        
        