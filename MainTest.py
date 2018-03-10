'''myarray = np.asarray(mylist)
np.array([[1,2,3],[4,5,6]]).tolist()

'''



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







