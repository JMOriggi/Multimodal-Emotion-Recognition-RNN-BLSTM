import os
import csv
import shutil
import numpy as np


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


def encodeLabels(arrayEmoLabel):
    i = 0
    emoEncoded = []
    while i < len(arrayEmoLabel):
        emoLabel = arrayEmoLabel[i]
        if  emoLabel == 'exc' or emoLabel == 'hap': 
            code = [1,0,0,0,0,0,0] #JOY
        if  emoLabel == 'ang' or emoLabel == 'fru':    
            code = [0,1,0,0,0,0,0] #ANG
        if  emoLabel == 'dis': 
            code = [0,0,1,0,0,0,0] 
        if  emoLabel == 'sad': 
            code = [0,0,0,1,0,0,0] 
        if  emoLabel == 'sur': 
            code = [0,0,0,0,1,0,0] 
        if  emoLabel == 'fea': 
            code = [0,0,0,0,0,1,0]  
        if  emoLabel == 'neu': 
            code = [0,0,0,0,0,0,1]
        if  emoLabel == 'other' or emoLabel == 'xxx': 
            code = [0,0,0,0,0,0,2]  #NOT CLASSIFIED
        emoEncoded.append(code)
        i += 1
        
    return emoEncoded


def saveEncLabelcsv(emoEncoded, arrayFileName, main_root):
    out_csv_labels_path = os.path.join(main_root+'\LablesEmotion')
    i = 0
    
    while i < len(emoEncoded):
        #CREATE OUTPUTS DATA FILE: remove if it already exist and recreate it new
        out_current_file = arrayFileName[i] + '.csv'
        out_current_file = os.path.join(out_csv_labels_path, out_current_file)
        try:
            os.remove(out_current_file)
        except OSError:
            pass
        
        #WRITE ON IT
        print(arrayFileName[i])
        print(emoEncoded[i])
        with open(out_current_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(emoEncoded[i])
        csvfile.close()
        i += 1


if __name__ == '__main__':
    
    #SET MAIN ROOT
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
     
    #READ DATAFILE AND BUILD ARRAYS
    arrayFileName, arrayEmoLabel = readDataFile(main_root)
    
    #ENCODE EMOTIONS LABELS
    emoEncoded = encodeLabels(arrayEmoLabel)
    
    #WRITE CSV FILE
    saveEncLabelcsv(emoEncoded, arrayFileName, main_root)
    
    print('END')
    
    