from keras.preprocessing.text import Tokenizer
import csv
import numpy as np

import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.clusterData(mainRoot)
trainData.encoder(mainRoot)

#READ Audio File Name 
datareader = csv.reader(open(os.path.join(mainRoot+'\AudioFilesName.csv'), 'r'))
data = []
for row in datareader:
    data.append(row)
X = data
print(X[0:10])
print(type(X))

#READ encoded emotion: Read the content as an array of numbers and not string as default
datareader = csv.reader(open(os.path.join(mainRoot+'\encodedEmo.csv'), 'r'))
data = []
for row in datareader:
    data.append([int(val) for val in row])
Y = np.array([np.array(xi) for xi in data])
print(Y[0:10])
print(type(Y))

#READ encoded text: Read the content as an array of numbers and not string as default
datareader = csv.reader(open(os.path.join(mainRoot+'\encodedText.csv'), 'r'))
data = []
for row in datareader:
    data.append([int(val) for val in row])
Z = np.array([np.array(xi) for xi in data])
print(Z[0:10])
print(type(Z))

