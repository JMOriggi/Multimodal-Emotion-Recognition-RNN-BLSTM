'''import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.moveCopyAudioFiles(mainRoot)'''

import numpy as np
import csv
from numpy import genfromtxt
import keras

'''print(a)
a = keras.preprocessing.sequence.pad_sequences(a, maxlen=5)
print(a)

print(b)
b = keras.preprocessing.sequence.pad_sequences(b, maxlen=5)
print(b)

x = []
x.append(a)
x.append(b)
x.append(a)
x.append(b)
print(x)
x = np.asarray(x)
print(x.shape)'''

''''x = []
a = np.asarray([[1,2,3],[4,5,6],[4,5,6]])
b = np.asarray([[7,8,9],[10,11,12]])
print(a.shape)
x.append(a)
x.append(b)
x.append(a)
x = np.asarray(x)
print('prepared array: ',x)

with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(x)
f.close()

datareader = csv.reader(open('output.csv', 'r'))
data = []
window = []
step = []
for row in datareader:
    for val in row:
        #print('val: ',val)
        val = val.split('[')[1]
        val = val.split(']')[0]
        len = val.count(' ')
        i = 0
        while i <= len: 
            step.append(int(val.split(' ')[i]))
            i +=1       
        window.append(np.array(step))
        step = []
    data.append(np.array(window))    
    window = []        
print('data: ', data) 
   
#Y = np.array([np.array(xi) for xi in data])

#Y = genfromtxt('output.csv', delimiter=',')
#print('Readed val: ',Y)'''

import os
import DataTrainingUtils as trainData

mainRootTraining = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')

trainData.buildAudioCsv(mainRootTraining)
trainData.readAudioCsv(mainRootTraining)




