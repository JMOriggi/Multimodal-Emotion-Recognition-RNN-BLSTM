import os
import csv
import shutil
import numpy as np


if __name__ == '__main__':
    
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_csv_labels_path = os.path.join(main_root+'\LablesEmotion') 
    
    #READ THE FILE AND BUILD ARRAYS
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
    
    #ENCODE EMOTIONS
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
    
    #WRITE CSV FILE
    #Emotion Labels
    i = 0
    while i < len(emoEncoded):
        #CREATE OUTPUTS DATA FILE: remove if it already exist and recreate it new
        out_current_file = arrayFileName[i] + '.csv'
        out_current_file = os.path.join(out_csv_labels_path, out_current_file) 
        print(out_current_file)
        try:
            os.remove(out_current_file)
        except OSError:
            pass
        
        with open(out_current_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(emoEncoded[i])
        csvfile.close()
        i += 1
    
    print('END')
    
    