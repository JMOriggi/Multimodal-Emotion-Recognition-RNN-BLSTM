##################################################################
#
#This function aim to compute and save audio features from audio
#wav files. 2 version of computing features have been implemented.
#The currently used and the better version, is V2. All features 
#extracted from an audio file will be saved in a csv file and  
#grouped by emotion class.
#
##################################################################


import os
import csv
import numpy as np
from scipy.io.wavfile import read
import librosa
import matplotlib.pyplot as plt 
import Compute_audio_featuresV1 as c_a_fV1
import Compute_audio_featuresV2 as c_a_fV2

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
Mode = 'All'
#Mode = 'test'
if Mode == 'All':
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_All')
if Mode == 'training':
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
if Mode == 'test':    
    main_root = os.path.normpath(r'DC:\DATA\POLIMI\----TESI-----\Corpus_Test')
all_wav_path = os.path.join(main_root + '\AllAudioFiles')
index_file_path =  os.path.join(main_root+'\AllData.txt')
out_audio_feature_path = os.path.join(main_root+'\FeaturesAudio')   
out_audio_feature_path_dirty = os.path.join(main_root+'\FeaturesAudio\Z_DataExt') 
    
# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

def saveFeaturecsv(currentFileFeatures, csvOutputFilePath):
    csvOutputFilePath = os.path.join(csvOutputFilePath + '.csv')
    try:
        os.remove(csvOutputFilePath)
    except OSError:
        pass
    with open(csvOutputFilePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.asarray(currentFileFeatures))
    f.close()


def readDataFile():
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    #Audio File Names
    with open(index_file_path, 'r') as AllDatafile:
        X = [line.strip() for line in AllDatafile] 
        arrayFileName = [line.split(';')[0] for line in X] 
    AllDatafile.close()
    #Emotion Labels
    with open(index_file_path, 'r') as AllDatafile:
        Y = [line.strip() for line in AllDatafile] 
        arrayEmoLabel = [line.split(';')[1] for line in Y]
    AllDatafile.close()  
    #Transcriptions   
    with open(index_file_path, 'r') as AllDatafile:
        Z = [line.strip() for line in AllDatafile] 
        arrayText = [line.split(';')[2] for line in Z]
    AllDatafile.close()
    
    return arrayFileName, arrayEmoLabel


def readWav(audioFilePath):
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    monoAudio = inputAudio[1]
    monoAudio = np.array(monoAudio, dtype=float) #type float necessary
    return monoAudio, sampleRate


def addBrownNoise(audioFilePath, BrownNoisePath, snr):
    x, sr = librosa.core.load(audioFilePath, mono=True)
    z, sr2 = librosa.core.load(BrownNoisePath, mono=True) 
    print('x.sr: ',sr)
    print('z.sr: ',sr2) 
    print('x.shape: ',x.shape)
    print('z.shape: ',z.shape)
    
    while z.shape[0] < x.shape[0]:  # loop in case noise is shorter than audio lenght
        z = np.concatenate((z, z), axis=0)
    z = z[0: x.shape[0]]
    print('z.shape2: ',z.shape)
    rms_z = np.sqrt(np.mean(np.power(z, 2)))
    rms_x = np.sqrt(np.mean(np.power(x, 2)))
    snr_linear = 10 ** (snr / 20.0)
    snr_linear_factor = rms_x / rms_z / snr_linear
    y = x + z * snr_linear_factor
    rms_y = np.sqrt(np.mean(np.power(y, 2)))
    y = y * rms_x / rms_y 
    print('y.shape: ',y.shape)
    return y

def computeFeaturesV1(arrayAudio, sampleRate):
    currentFileFeatures = c_a_fV1.compute_features(arrayAudio, sampleRate)
    #print(currentFileFeatures)
    print(currentFileFeatures.shape)
    return currentFileFeatures

def computeFeaturesV2(arrayAudio, sampleRate):
    currentFileFeatures = c_a_fV2.compute_features(arrayAudio, sampleRate, False)
    #print(currentFileFeatures)
    print(currentFileFeatures.shape)
    return currentFileFeatures


def buildAudioFeaturesCsv(arrayEmoLabel, audioDirectoryPath, out_audio_feature_path, out_audio_feature_path_dirty, dataExt_flag):
    currentFileFeatures = []
    audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
    i = 0
    for audioFile in audlist:
        print('ROUND: ',i,'/',len(audlist))
        print(audioFile)
        
        #TAKE CORRECT DIRECTORY PATH
        direc = 'oth'
        if  arrayEmoLabel[i] == 'exc': 
            direc = 'joy' #JOY
        if  arrayEmoLabel[i] == 'ang':    
            direc = 'ang' #ANG
        if  arrayEmoLabel[i] == 'sad': 
            direc = 'sad'  
        if  arrayEmoLabel[i] == 'neu': 
            direc = 'neu'  
        
        #CURRENT FILE FEATURE
        if dataExt_flag == False:
            #Compute
            audioFilePath = os.path.join(audioDirectoryPath, audioFile)
            arrayAudio, sampleRate = readWav(audioFilePath)
            currentFileFeatures = computeFeaturesV2(arrayAudio, sampleRate)
            #Save
            csvOutputFilePath = os.path.join(out_audio_feature_path, direc)
            csvOutputFilePath = os.path.join(csvOutputFilePath, audioFile.split('.')[0])
            saveFeaturecsv(currentFileFeatures, csvOutputFilePath)
        
        #SAVE DIRTY COPY OF THE FILE FOR DATA EXTENSION: only for training data
        if dataExt_flag == True:
            #Mix
            BrownNoisePath = os.path.join(out_audio_feature_path_dirty, 'BrownNoise.wav')
            snr = 18   
            arrayAudio_dirty = addBrownNoise(audioFilePath, BrownNoisePath, float(snr))
            #Compute
            currentFileFeatures_dirty = computeFeaturesV2(arrayAudio_dirty, sampleRate)
            print('sampleRate',sampleRate)
            #Save
            csvOutputFilePath_dirty = os.path.join(out_audio_feature_path_dirty, direc)
            csvOutputFilePath_dirty = os.path.join(csvOutputFilePath_dirty, audioFile.split('.')[0]+'_dirty')
            saveFeaturecsv(currentFileFeatures_dirty, csvOutputFilePath_dirty)
        
        currentFileFeatures = []
        i += 1
    
    
if __name__ == '__main__':
        
    #True compute only feature of BN audio mix
    dataExt_flag = False
    
    #READ DATAFILE AND BUILD ARRAYS
    arrayFileName, arrayEmoLabel = readDataFile()
    
    buildAudioFeaturesCsv(arrayEmoLabel, all_wav_path, out_audio_feature_path, out_audio_feature_path_dirty, dataExt_flag)  
        
    print('****END')
     
