'''import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.moveCopyAudioFiles(mainRoot)'''

import numpy as np
import csv

a = [[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]]
x = []
'''x.append(np.asarray(a))
x.append(np.asarray(a))
x.append(np.asarray(a))'''
x.append(a)
x.append(a)
x.append(a)
print(x)

with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(x)
f.close()

datareader = csv.reader(open('output.csv', 'r'))
data = []
for row in datareader:
    data.append([val for val in row])
print(data)    
Y = np.array([np.array(xi) for xi in data])
print(Y)