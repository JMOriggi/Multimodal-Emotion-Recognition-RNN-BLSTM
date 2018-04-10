import csv
import os
import sys
import gensim
import numpy as np


def get_text_bynary(text, model, notFoundCounter, notFoundedWord):
    
    encodedText = []
    
    print(text)
    for word in text:
        #REMOVE PUNCTUATION
        word = ''.join(e for e in word if e.isalnum() or e == "'")
        
        try:
            encodedText.append(model.wv[word])
        except KeyError:
            print('NOT FOUND')
            encodedText.append(model.wv['unknown'])
            notFoundedWord.append(word)
            notFoundCounter += 1
    
    print(np.asarray(encodedText).shape)
    
    #TRANFORM ENCODED TEXT
    #encodedText = np.array(encodedText, np.float32)
    #encodedText = encodedText.tobytes()
    
    return encodedText, notFoundCounter, notFoundedWord


def readDatacsv(index_file_path):
    #Audio File Names
    with open(index_file_path, 'r') as AllDatafile:
        X = [line.strip() for line in AllDatafile] 
        arrayFileName = [line.split(';')[0] for line in X] 
    AllDatafile.close()
    #Transcriptions   
    with open(index_file_path,  encoding='utf-8') as AllDatafile:
        Z = [line.strip() for line in AllDatafile] 
        arrayText = [line.split(';')[2] for line in Z]
    AllDatafile.close()
    
    return arrayFileName, arrayText


def saveFeaturecsv(currentFilename, out_text_feature_path):
    csvOutputFilePath = os.path.join(out_text_feature_path, currentFilename)
    csvOutputFilePath = csvOutputFilePath + '.csv'
    try:
        os.remove(csvOutputFilePath)
    except OSError:
        pass
    
    with open(csvOutputFilePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.asarray(text_byn))
    f.close()


if __name__ == '__main__':

    #SET ROOT
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    #main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
    
    #SET PATHS
    modelPath = os.path.join(main_root+'\W2V_model\glove_WIKI')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_text_feature_path = os.path.join(main_root+'\FeaturesText') 
    
    #READ THE FILE AND BUILD ARRAYS
    arrayFileName, arrayText = readDatacsv(index_file_path)
    
    #SET MODEL AND PARAMETERS
    notFoundCounter = 0
    notFoundedWord = []
    model = gensim.models.Word2Vec.load(modelPath)#Se errore pip install gensim==3.0

    #FOR EACH FILE ENCODE TEXT
    i = 0
    while i < len(arrayText):
        print(arrayFileName[i])
        
        #TRANSFORM TEXT
        current_text = arrayText[i]
        current_text = arrayText[i].lower() #Lower case
        current_text = current_text.split() #Isolate words
        
        #ENCODE
        text_byn, notFoundCounter, notFoundedWord = get_text_bynary(current_text, model, notFoundCounter, notFoundedWord)
        
        #WRITE OUTPUT FILE
        saveFeaturecsv(arrayFileName[i], out_text_feature_path)
        
        i += 1
    
    print('notFoundCounter: ', notFoundCounter)
    print('notFoundedWord: ', notFoundedWord)
       
    print('END')   
