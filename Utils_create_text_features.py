import glob
import os
import sys
import gensim
import numpy as np
import tensorflow as tf


def write_to_tfrecord(label, binary_text, writer):
    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_text]))
    }))

    writer.write(example.SerializeToString())


def get_text_bynary(text, model):
    
    print('get_text_bynary')
    
    counter = 0
    translated_text = []
    notfounded = []

    for word in text:
        print(word)
        try:
            translated_text.append(model.wv[word])
        except KeyError:
            translated_text.append(model.wv['unknown'])
            notfounded.append(word)

    #text = np.array(translated_text, np.float32)
    #return text.tobytes(), notfounded, counter
    return text, notfounded, counter

if __name__ == '__main__':

    #main_root = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\NewCorpus')
    main_root = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
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
    
    print(arrayText)
    
    counter = 0
    modelPath = os.path.join(main_root+'\W2V_model\glove_WIKI')
    model = gensim.models.Word2Vec.load(modelPath)#Se errore pip install gensim==3.0

    # iterate over the list getting each file
    ''''i = 0
    while i < len(arrayFileName):
        current_text = arrayText[i]
        text_byn, nf, cnt = get_text_bynary(current_text, model)
        print(text_byn)
        i += 1
    '''    
    print('END')   
