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

def readWav(audioFilePath):
    #READ THE WAV FILE
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    stereoAudio = inputAudio[1]
    #TRASFORM IN MONO: no need because audio already in mono
    monoAudio = stereoAudio
    monoAudio = np.array(monoAudio, dtype=float) #type float necessary
    return monoAudio, sampleRate

def buildFeatures(monoAudio, sampleRate):
    
    #n_fft is the window of 25ms = 400samples e hop_lenght is the step between one frame and another
    mfcc = librosa.feature.mfcc(monoAudio, sampleRate, n_mfcc=40, hop_length=int(0.010*sampleRate), n_fft=int(0.020*sampleRate))
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    print(mfcc.shape)
    print(mfcc_delta.shape)
    print(mfcc_delta_delta.shape)
    #Transpose because librosa gives me the freq in the row but i want them in the collums
    mfcc = mfcc.T
    mfcc_delta = mfcc_delta.T
    mfcc_delta_delta = mfcc_delta_delta.T
    print(mfcc.shape)
    print(mfcc_delta.shape)
    print(mfcc_delta_delta.shape)
    
    pitch_and_intensity.extractPitch(wavFN, outputFN, praatEXE, minPitch, maxPitch, sampleStep, silenceThreshold, forceRegenerate, undefinedValue, medianFilterWindowSize, pitchQuadInterp)
    pitch_and_intensity.extractIntensity(inputFN, outputFN, praatEXE, minPitch, sampleStep, forceRegenerate, undefinedValue)
    
    #to concatenate row per row 
    X = np.hstack((mfcc, mfcc_delta, mfcc_delta_delta))
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
        
        
        