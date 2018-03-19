
#TEST DATA TRAINING CLASS
#from DataTrainingUtils import DataTrainingUtils
#trainData = DataTrainingUtils()
#trainData.setDataCorpus()
#output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
#print('---Coresponding output for Audio Ses04F_script01_1_M019---')
#print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text)            
 
#TEST AUDIO CLASS     
'''import AudioUtils as aud
import NeuralNetworkUtils as nn
import numpy as np
audioFileName ='file.wav'
arrayAudio, sampleRate = aud.getArrayFromAudio(audioFileName)
allFrame = aud.getFrameArray(arrayAudio, sampleRate, 1024)
allFrameFFT = aud.getSpectrumFrameArray(allFrame)'''
#nn.FFNNModel(np.float32(allFrameFFT[0]), 1)
'''import numpy as np
from keras.models import Sequential
from keras.layers import Dense
data = list(range(0,99))
labels = list(range(1,100))
print('Data: ',data)
print('labels: ',labels)
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data,labels,epochs=10,batch_size=32)
predictions = model.predict(data)
print('predictions: ',predictions)'''

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

sample_rate, samples = wavfile.read('Ses03F_impro08_M003.wav')
print('samples: ', samples)
print('Samples shape: ', samples.shape)
print('Samples duration: ', len(samples)/sample_rate)
print('sample_rate: ', sample_rate)

fft, freqsBins, timeBins, im = plt.specgram(samples, Fs=sample_rate, NFFT=320, cmap=plt.get_cmap('autumn_r'))
print('Pxx: ', fft)
print('len Pxx: ', len(fft))
print('len2 Pxx: ', len(fft[0]))
print('shape Pxx: ', fft.shape)
print('shape freqs: ', freqsBins.shape)
print('shape bins: ', timeBins.shape)

cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()

i = 0
X = np.full((len(fft[0]), len(fft)), 0)
while i < len(fft):
    y = 0
    while y < len(fft[0]):
        X[y][i] = fft[i][y]
        y+=1
    i+=1
print('New shape Pxx: ', X.shape)
print('New Pxx: ', X)







