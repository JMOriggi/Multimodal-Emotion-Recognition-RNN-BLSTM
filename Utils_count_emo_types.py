import os
import csv
import shutil
import numpy as np
import AudioUtils as aud

def countLabels(arrayEmoLabel):
    counter = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
    i = 0
    
    while i < len(arrayEmoLabel):
        emoLabel = arrayEmoLabel[i]
        if  emoLabel == 'exc' or emoLabel == 'hap': 
            counter[0] += 1
        if  emoLabel == 'ang' or emoLabel == 'fru':    
            counter[1] += 1
        if  emoLabel == 'dis': 
            counter[2] += 1
        if  emoLabel == 'sad': 
            counter[3] += 1 
        if  emoLabel == 'sur': 
            counter[4] += 1 
        if  emoLabel == 'fea': 
            counter[5] += 1  
        if  emoLabel == 'neu': 
            counter[6] += 1
        if  emoLabel == 'other' or emoLabel == 'xxx': 
            counter[7] += 1
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
    #main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
    
    arrayFileName, arrayEmoLabel = readDataFile(main_root) 
    counter = countLabels(arrayEmoLabel)
    print(counter)
    print('END')
    