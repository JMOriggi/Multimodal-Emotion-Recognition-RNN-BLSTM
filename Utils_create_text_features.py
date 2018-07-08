##################################################################
#
#This function aim to encode all the transcription of audio files,
#and in this way create the text feature to feed the NN. The 
#Word2Vec encoding is used to give a vector representation for
#each word in the text file.
#
##################################################################


import csv
import os
import gensim #Se errore pip install gensim==3.0
import numpy as np

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
Mode = 'training'
#Mode = 'test'
if Mode == 'training':
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
if Mode == 'test':    
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Test')
modelPath = os.path.join(main_root+'\W2V_model\glove_WIKI')
index_file_path =  os.path.join(main_root+'\AllData.txt')
out_text_feature_path = os.path.join(main_root+'\FeaturesText')  
  
# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

def saveFeaturecsv(featureEncoded, arrayFileName, emoFolder):
    out_csv_labels_path = os.path.join(out_text_feature_path, emoFolder)
    i = 0
    
    while i < len(featureEncoded):
        #CREATE OUTPUTS DATA FILE: remove if it already exist and recreate it new
        out_current_file = arrayFileName[i] + '.csv'
        out_current_file = os.path.join(out_csv_labels_path, out_current_file)
        try:
            os.remove(out_current_file)
        except OSError:
            pass
        
        #WRITE ON IT
        with open(out_current_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(featureEncoded[i])
        csvfile.close()
        i += 1


def readData():
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
    with open(index_file_path,  encoding='utf-8') as AllDatafile:
        Z = [line.strip() for line in AllDatafile] 
        arrayText = [line.split(';')[2] for line in Z]
    AllDatafile.close()
    
    return arrayFileName, arrayText, arrayEmoLabel


def get_text_bynary(text, model, notFoundCounter, notFoundedWord):
    
    encodedText = []
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
    
    #PRINT
    '''print(text)
    print(np.asarray(encodedText).shape)'''
    
    #TRANFORM ENCODED TEXT
    '''encodedText = np.array(encodedText, np.float32)
    encodedText = encodedText.tobytes()'''
    
    return encodedText, notFoundCounter, notFoundedWord


def encodeText(arrayFileName, arrayText, arrayEmoLabel):
    
    #SET MODEL AND PARAMETERS
    notFoundCounter = 0
    notFoundedWord = []
    model = gensim.models.Word2Vec.load(modelPath)
    joyEncoded = []
    sadEncoded = []
    angEncoded = []
    neuEncoded = []
    joyFileName = []
    sadFileName = []
    angFileName = []
    neuFileName = []
        
    #FOR EACH FILE ENCODE TEXT
    i = 0
    while i < len(arrayText):
        print(arrayFileName[i])
        emoLabel = arrayEmoLabel[i]
        
        #TRANSFORM TEXT
        current_text = arrayText[i]
        current_text = arrayText[i].lower() #Lower case
        current_text = current_text.split() #Isolate words
        
        #ENCODE
        text_byn, notFoundCounter, notFoundedWord = get_text_bynary(current_text, model, notFoundCounter, notFoundedWord)
            
        #APPEND FOR EACH EMOTION    
        if emoLabel == 'exc':
            joyEncoded.append(text_byn)
            joyFileName.append(arrayFileName[i])
        if emoLabel == 'ang':   
            angEncoded.append(text_byn)
            angFileName.append(arrayFileName[i])
        if emoLabel == 'sad':    
            sadEncoded.append(text_byn)
            sadFileName.append(arrayFileName[i])
        if emoLabel == 'neu':   
            neuEncoded.append(text_byn)
            neuFileName.append(arrayFileName[i])
        
        i += 1
    
    print('notFoundCounter: ', notFoundCounter)
    print('notFoundedWord: ', notFoundedWord)
    
    return joyEncoded, angEncoded, sadEncoded, neuEncoded, joyFileName, angFileName, sadFileName, neuFileName


if __name__ == '__main__':
    
    #READ THE FILE AND BUILD ARRAYS
    arrayFileName, arrayText, arrayEmoLabel = readData()
    
    #ENCODE EMOTIONS LABELS
    joyEncoded, angEncoded, sadEncoded, neuEncoded, joyFileName, angFileName, sadFileName, neuFileName = encodeText(arrayFileName, arrayText, arrayEmoLabel)
    
    #WRITE CSV FILE
    saveFeaturecsv(joyEncoded, joyFileName, 'joy')
    saveFeaturecsv(angEncoded, angFileName, 'ang')
    saveFeaturecsv(sadEncoded, sadFileName, 'sad')
    saveFeaturecsv(neuEncoded, neuFileName, 'neu')
      
    print('END')   
