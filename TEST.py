'''import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.moveCopyAudioFiles(mainRoot)'''

import numpy as np
import csv
from numpy import genfromtxt
import keras

a = np.asarray([[1,2,3],[4,5,6]])
b = np.asarray([[7,8,9],[10,11,12]])
print(a.shape)

print(a)
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
print(x.shape)

''''x = []
x.append(a)
x.append(b)
x.append(a)
x.append(b)
print(x)

x = np.asarray(x)
print(x.shape)'''


'''x.append(np.asarray(a))
x.append(np.asarray(a))
x.append(np.asarray(a))
x.append(a)
x.append(a)
x.append(a)
print('prepared array: ',x)

with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(x)
f.close()

datareader = csv.reader(open('output.csv', 'r'))
data = []
for row in datareader:
    data.append([np.array(val) for val in row])
print(data)    
Y = np.array([np.array(xi) for xi in data])

Y = genfromtxt('output.csv', delimiter=',')
print('Readed val: ',Y)'''