import os
import csv
import shutil
import numpy as np


def countLabels(arrayEmoLabel):
    counter = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    i = 0
    while i < len(arrayEmoLabel):
        emoLabel = arrayEmoLabel[i]
        if  emoLabel == 'exc': 
            counter[0] += 1 #JOY
        if  emoLabel == 'ang':    
            counter[1] += 1 #ANG
        if  emoLabel == 'sad': 
            counter[2] += 1    
        if  emoLabel == 'neu': 
            counter[3] += 1  
        if  emoLabel == 'sur': 
            counter[4] += 1 
        if  emoLabel == 'fea': 
            counter[5] += 1  
        if  emoLabel == 'hap': 
            counter[6] += 1
        if  emoLabel == 'dis': 
            counter[7] += 1
        if  emoLabel == 'fru': 
            counter[8] += 1    
        if  emoLabel == 'oth' or emoLabel == 'xxx': 
            counter[9] += 1
        i += 1
        
    return counter


def countLabelsV2(arrayEmoLabel):
    counter = np.array([[0],[0],[0],[0],[0]])
    i = 0
    
    while i < len(arrayEmoLabel):
        emoLabel = arrayEmoLabel[i]
        if  emoLabel == 'exc': 
            counter[0] += 1 #JOY
        if  emoLabel == 'ang':    
            counter[1] += 1 #ANG
        if  emoLabel == 'sad': 
            counter[2] += 1    
        if  emoLabel == 'neu': 
            counter[3] += 1  
        if  emoLabel == 'oth' or emoLabel == 'xxx' or emoLabel == 'hap' or emoLabel == 'fru' or emoLabel == 'sur' or emoLabel == 'fea' or emoLabel == 'dis': 
            counter[4] += 1  #NOT CLASSIFIED
        i += 1
        
    return counter


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


if __name__ == '__main__':
    
    #SET MAIN ROOT
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Training')
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
    #main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Usefull')
    
    arrayFileName, arrayEmoLabel = readDataFile(main_root) 
    counter = countLabels(arrayEmoLabel)
    print('V1: ',counter)
    counter = countLabelsV2(arrayEmoLabel)
    print('V2: ',counter)
    print('END')
    