import numpy as np
import csv
from numpy import genfromtxt
import os
import librosa
import operator


    
x = np.array([0.2,0.01,0.3,0.001])
index, value = max(enumerate(x), key=operator.itemgetter(1))
print(index, ' ', value)
x = np.array([0.2,0.6,0.3,0.001])
index, value = max(enumerate(x), key=operator.itemgetter(1))
print(index, ' ', value)
x = np.array([0.2,0.6,0.3,0.001])
index, value = max(enumerate(x), key=operator.itemgetter(1))
print(index, ' ', value)
x = np.array([0.2,0.6,0.3,0.001])
index, value = max(enumerate(x), key=operator.itemgetter(1))
print(index, ' ', value)

'''all_wav_path = os.path.join(main_root + '\AllAudioFiles')
audioDirectoryPath = os.path.normpath(main_root + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
print(audlist)'''

def test():
    #mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training\FeaturesText')
    mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus\FeaturesAudio')
    list = [ item for item in os.listdir(mainRoot) if os.path.isfile(os.path.join(mainRoot, item)) ]
    
    #READ encoded emotion: Read the content as an array of numbers and not string as default
    for file in list:
        datareader = csv.reader(open(os.path.join(mainRoot,file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        print(Y.shape)
     
    print('END') 