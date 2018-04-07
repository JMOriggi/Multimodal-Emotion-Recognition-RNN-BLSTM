import numpy as np
import csv
from numpy import genfromtxt
import keras
import os
import DataTrainingUtils as trainData
import librosa





'''main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
all_wav_path = os.path.join(main_root + '\AllAudioFiles')
audioDirectoryPath = os.path.normpath(main_root + '\AllAudioFiles')
audlist = [ item for item in os.listdir(audioDirectoryPath) if os.path.isfile(os.path.join(audioDirectoryPath, item)) ]
    
print(audlist)'''
