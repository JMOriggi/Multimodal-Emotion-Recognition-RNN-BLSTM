from keras.preprocessing.text import Tokenizer
import csv
import numpy as np
import csv
# define 5 documents
'''docs = ['Well done! Well done',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!','klass']'''
'''docs = ['Well done! Well done Great ! pussy']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_index)
#X = t.texts_to_sequences(docs)
#print(X)
#Xdata= []

docs2 = ['Well done! Well done','Great', 'pussy']
#t.fit_on_texts(docs2)
# summarize what was learned
#print(t.word_index)
X = t.texts_to_sequences(docs2)
print(X)'''

'''thefile = open('testttt.txt', 'w')
for item in X:
  thefile.write("%s\n" % item)
thefile.close()

with open('testttt.txt', 'r') as f:
    for line in f:
        print(line)
Xdata.append(line)
print(Xdata)'''

'''with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(X)

datafile = open('output.csv', 'r')
datareader = csv.reader(datafile)
data = []
for row in datareader:
    data.append([int(val) for val in row])
print(data)
Y = np.array([np.array(xi) for xi in data])
print(Y)
print(type(Y))
print(Y[0])'''

import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.buildDictionary(mainRoot)
trainData.encodeText(mainRoot)

datafile = open('output.csv', 'r')
datareader = csv.reader(datafile)
data = []
for row in datareader:
    data.append([int(val) for val in row])
print(data)

Y = np.array([np.array(xi) for xi in data])
print(Y)
print(type(Y))
print(Y[0])
