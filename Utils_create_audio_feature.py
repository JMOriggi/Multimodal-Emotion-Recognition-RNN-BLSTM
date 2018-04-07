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


def readWav(audioFilePath):
    #READ THE WAV FILE
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    stereoAudio = inputAudio[1]
    #TRASFORM IN MONO: no need because audio already in mono
    monoAudio = stereoAudio
    monoAudio = np.array(monoAudio, dtype=float)
    return monoAudio, sampleRate

def buildFeature(monoAudio, sampleRate): 
    #COMPUTE SPECTROGRAM: NFFT=how many sample in one chunk, for Fs16000 chunks20ms-->32
    fft, freqsBins, timeBins, im = plt.specgram(monoAudio, Fs=sampleRate, NFFT=320, cmap=plt.get_cmap('autumn_r'))
    
    #RESHAPE FREQ ARRAY: row=timestep collumns=freq values
    maxFftValues = 50 #len(fft) #voce si estende 80Hz-1500Hz mentre fft va da 0-8000 (con 50 ho fino a 2450Hz)
    X = np.full((len(fft[0]), maxFftValues), 0)
    i = 0
    while i < maxFftValues:
        y = 0
        while y < len(fft[0]):
            X[y][i] = fft[i][y]
            y+=1
        i+=1
    
    #n_fft is the window of 25ms = 400samples e hop_lenght is the step between one frame and another
    mfcc = librosa.feature.mfcc(monoAudio, sampleRate, n_mfcc=40, hop_length=int(0.010*sampleRate), n_fft=int(0.020*sampleRate))
    print(mfcc)
    print(mfcc.dtype)
    print(mfcc.shape)
    #Transpose because librosa gives me the freq in the row but i want them in the collums
    mfcc = mfcc.T
    print(mfcc)
    print(mfcc.shape)
    
    #to concatenate row per row 
    x = np.hstack((mfcc,mfcc,mfcc))
    print(x.shape)
    
    return mfcc

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
        
        currentFileFeatures = buildFeature(arrayAudio, sampleRate)
        
        saveFeaturecsv(currentFileFeatures, csvOutputFilePath)
        
        currentFileFeatures = []
        
    
    
if __name__ == '__main__':
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    all_wav_path = os.path.join(main_root + '\AllAudioFiles')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_audio_feature_path = os.path.join(main_root+'\FeaturesAudio')   
    
    buildAudioCsv(index_file_path, all_wav_path, out_audio_feature_path)  
        
    print('****END')
        
        
        