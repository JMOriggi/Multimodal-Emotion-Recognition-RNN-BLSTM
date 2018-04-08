import glob
import os
import sys
import gensim
import numpy as np
import tensorflow as tf

mainRoot = 'D:\DATA\POLIMI\----TESI-----\NewCorpus' #Cartella madre
text_path = os.path.join(mainRoot+'\allData.txt') #file indice dei dati: nomeFile;emo;transcrizione
outfile_Path_Tfr = '/Users/Mauro/Desktop/root_path_for_doubled_dataset/out/text/TestSet/Smoothed.tfrecords'


def write_to_tfrecord(label, binary_text, writer):
    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_text]))
    }))

    writer.write(example.SerializeToString())


def get_text_bynary(text, label, model):
    counter = 0
    translated_text = []
    lab = []
    notfounded = []

    for word in text:
        if word == 'è':
            word = word.replace('è', "e'")
        counter += 1
        try:
            translated_text.append(model.wv[word])
        except KeyError:
            translated_text.append(model.wv['unknown'])
            notfounded.append(word)

    lab.append(label)
    text = np.array(translated_text, np.float32)
    label = np.array(lab, np.float32)
    return text.tobytes(), label.tobytes(), notfounded, counter


def make_label(file):
    (head, _) = os.path.split(file)
    (_, label) = os.path.split(head)

    print(label)

    # 001, questions, 010 exclamations, 100 statements
    if label == 'Questions':
        one_hot_lab = [0, 0, 1]
    elif label == 'Statements':
        one_hot_lab = [1, 0, 0]
    elif label == 'Exclamations':
        one_hot_lab = [0, 1, 0]
    else:
        raise ("the fuck is goin' on")

    return (one_hot_lab, label)


if __name__ == '__main__':

    local_path = os.path.dirname(__file__)

    # take a list of folder path 
    args = [text_path, outfile_Path_Tfr]

    counter = 0
    notf = []

    model = gensim.models.Word2Vec.load(local_path + '/W2V_model/glove_WIKI')
    files = glob.glob(args[0])
    writer = tf.python_io.TFRecordWriter(args[3])

    # iterate over the list getting each file

    for file in files:
        # open the file and then call .read() to get the text
        with open(file, encoding='utf-8') as f:
            (one_hot_lab, label) = make_label(file)
            print('procesing file name:', file, 'Label:', label, 'One Hot Repr:', one_hot_lab)
            text = f.read().split()
            text_byn, oh_lab_byn, nf, cnt = get_text_bynary(text, one_hot_lab, model)
            counter += cnt
            if nf:
                notf.append(nf)
            if len(text_byn) != 0:
                write_to_tfrecord(oh_lab_byn, text_byn, writer)
    writer.close()

    print("tot word:", counter)
    print("words not found:", len(notf))
    print(notf)
