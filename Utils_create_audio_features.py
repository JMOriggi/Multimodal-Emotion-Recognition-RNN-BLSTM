import os
import csv
import numpy as np
from scipy.io.wavfile import read
import librosa

def estimate_pitch(monoAudio, sampleRate, window, hop):
    pitch , mag = librosa.core.piptrack(monoAudio, sampleRate, n_fft=window, hop_length=hop, fmin=50, fmax=1500)
    pitch = pitch.T
    mag = mag.T
    
    f0 = []
    i = 0
    while i < len(pitch):
        index = mag[i, :].argmax()
        f0.append(pitch[i, index])
        i += 1
    
    f0 = np.asarray(f0)
    f0_delta = librosa.feature.delta(f0, width=3)
    f0_delta_delta = librosa.feature.delta(f0, width=3, order=2)
    
    f0 = f0.reshape((len(f0),1))
    f0_delta = f0_delta.reshape((len(f0),1))
    f0_delta_delta = f0_delta_delta.reshape((len(f0),1))
    
    '''print(f0.shape)
    print(f0_delta.shape)
    print(f0_delta_delta.shape)'''
    
    return f0, f0_delta, f0_delta_delta


def readWav(audioFilePath):
    inputAudio = read(audioFilePath)
    sampleRate = inputAudio[0]
    monoAudio = inputAudio[1]
    monoAudio = np.array(monoAudio, dtype=float) #type float necessary
    return monoAudio, sampleRate


def computeFeatures(monoAudio, sampleRate):
    #TODO: togliere primo e ultimo frame per tutte le feature
    
    #SET PARAMETERS
    window = int(0.020*sampleRate)
    hop = int(0.010*sampleRate)
    mfcc_coef_size = 12
    
    #STFT
    D = librosa.core.stft(monoAudio, n_fft=window, hop_length=hop)
    stft = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)
    stft = stft.T
    print(np.asarray(stft).shape)
    
    #ZERO CORSSING RATE
    zero_crossing = librosa.feature.zero_crossing_rate(monoAudio, frame_length=window, hop_length=hop)
    zero_crossing_delta = librosa.feature.delta(zero_crossing, width=3)
    
    #PITCH:
    pitch, pitch_delta, pitch_delta_delta = estimate_pitch(monoAudio, sampleRate, window, hop)
    
    #ENERGY
    energy = librosa.feature.rmse(y=monoAudio, frame_length=window, hop_length=hop)
    energy_delta = librosa.feature.delta(energy, width=3)
    energy_delta_delta = librosa.feature.delta(energy, width=3, order=2)
    
    #MFCC
    #n_fft is the window of 25ms = 400samples e hop_lenght is the step between one frame and another
    #For the deltas important to set width=3 to compute derivative of each sample with previous and next sample.
    mfcc = librosa.feature.mfcc(monoAudio, sampleRate, n_mfcc=mfcc_coef_size, n_fft=window, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
    
    #MFCC ENERGY
    mfcc_energy = librosa.feature.rmse(S=mfcc, frame_length=window, hop_length=hop)
    mfcc_energy_delta = librosa.feature.rmse(S=mfcc_delta, frame_length=window, hop_length=hop)
    mfcc_energy_delta_delta = librosa.feature.rmse(S=mfcc_delta_delta, frame_length=window, hop_length=hop)
    
    #TRASPOSE: librosa gives me the freq in the row but i want them in the collums
    zero_crossing = zero_crossing.T
    zero_crossing_delta = zero_crossing_delta.T
    energy = energy.T
    energy_delta = energy_delta.T
    energy_delta_delta = energy_delta_delta.T
    mfcc = mfcc.T
    mfcc_delta = mfcc_delta.T
    mfcc_delta_delta = mfcc_delta_delta.T
    mfcc_energy = mfcc_energy.T
    mfcc_energy_delta = mfcc_energy_delta.T
    mfcc_energy_delta_delta = mfcc_energy_delta_delta.T
    '''print('zero_crossing.shape: ', zero_crossing.shape)
    print('mfcc.shape: ', mfcc.shape)
    print('mfcc_delta.shape: ', mfcc_delta.shape)
    print('mfcc_delta_delta.shape: ', mfcc_delta_delta.shape)
    print('energy.shape: ', energy.shape)
    print('energy_delta.shape: ', energy_delta.shape)
    print('energy_delta_delta.shape: ', energy_delta_delta.shape)
    print('mfcc_energy.shape: ', mfcc_energy.shape)
    print('mfcc_energy_delta.shape: ', mfcc_energy_delta.shape)
    print('mfcc_energy_delta_delta.shape: ', mfcc_energy_delta_delta.shape)'''
    
    #CONCATENATE FEATURES PER ROWS
    #X = np.hstack((pitch, pitch_delta, pitch_delta_delta, energy, energy_delta, energy_delta_delta, mfcc, mfcc_delta, mfcc_delta_delta, mfcc_energy, mfcc_energy_delta, mfcc_energy_delta_delta))
    X = np.hstack((stft, pitch, pitch_delta, energy, energy_delta, zero_crossing, zero_crossing_delta, mfcc, mfcc_delta))
    print(X.shape) #Features final shape for current audio file
    
    return X


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


def readDataFile(main_root):
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


def buildAudioFeaturesCsv(arrayEmoLabel, audioDirectoryPath, out_audio_feature_path):
    currentFileFeatures = []
    audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
    
    i = 0
    for audioFile in audlist:
        print(i,'/',len(audlist))
        print(audioFile)
        
        audioFilePath = os.path.join(audioDirectoryPath, audioFile)
        
        arrayAudio, sampleRate = readWav(audioFilePath)
        
        currentFileFeatures = computeFeatures(arrayAudio, sampleRate)
        
        direc = 'oth'
        if  arrayEmoLabel[i] == 'exc' or arrayEmoLabel[i] == 'hap': 
            direc = 'joy' #JOY
        if  arrayEmoLabel[i] == 'ang' or arrayEmoLabel[i] == 'fru':    
            direc = 'ang' #ANG
        if  arrayEmoLabel[i] == 'sad': 
            direc = 'sad'  
        if  arrayEmoLabel[i] == 'neu': 
            direc = 'neu'  
            
        csvOutputFilePath = os.path.join(out_audio_feature_path, direc)
        csvOutputFilePath = os.path.join(csvOutputFilePath, audioFile.split('.')[0])
        saveFeaturecsv(currentFileFeatures, csvOutputFilePath)
        
        currentFileFeatures = []
        
        i += 1
    
    
if __name__ == '__main__':
    
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    #main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Training')
    #main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Usefull')
    
    all_wav_path = os.path.join(main_root + '\AllAudioFiles')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_audio_feature_path = os.path.join(main_root+'\FeaturesAudio')   
    
    
    #READ DATAFILE AND BUILD ARRAYS
    arrayFileName, arrayEmoLabel = readDataFile(main_root)
    
    
    buildAudioFeaturesCsv(arrayEmoLabel, all_wav_path, out_audio_feature_path)  
        
    print('****END')
        
        
        