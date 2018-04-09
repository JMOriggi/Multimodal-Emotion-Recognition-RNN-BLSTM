import glob
import os
import sys
import gensim
import numpy as np
import tensorflow as tf
import string


def write_to_tfrecord(label, binary_text, writer):
    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_text]))
    }))

    writer.write(example.SerializeToString())


def get_text_bynary(text, model, notFoundCounter):
    
    
    print(text)
    
    encodedText = []
    notFoundedWord = []

    for word in text:
        #Remove punctuation
        word = ''.join(e for e in word if e.isalnum())
        print(word)
        try:
            encodedText.append(model.wv[word])
        except KeyError:
            print('NOT FOUND')
            encodedText.append(model.wv['unknown'])
            notFoundedWord.append(word)
    
    print('translated_text shape: ',np.asarray(encodedText).shape)
    #TRANFORM ENCODED TEXT
    encodedText = np.array(encodedText, np.float32)
    encodedText = encodedText.tobytes()
    
    
    return encodedText, notFoundCounter, notFoundedWord

if __name__ == '__main__':

    #SET ROOTS
    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
    modelPath = os.path.join(main_root+'\W2V_model\glove_WIKI')
    index_file_path =  os.path.join(main_root+'\AllData.txt')
    out_text_feature_path = os.path.join(main_root+'\FeaturesText') 
    
    #READ THE FILE AND BUILD ARRAYS
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
    
    notFoundCounter = 0
    model = gensim.models.Word2Vec.load(modelPath)#Se errore pip install gensim==3.0

    #FOR EACH FILE ENCODE TEXT
    i = 0
    while i < len(arrayFileName):
        print(arrayFileName[i])
        
        #Lower case
        current_text = arrayText[i].lower()
        #Isolate words
        current_text = current_text.split()
        
        text_byn, notFoundCounter, notFoundedWord = get_text_bynary(current_text, model, notFoundCounter)

        i += 1
    
    print('notFoundCounter: ', notFoundCounter)
       
    print('END')   
