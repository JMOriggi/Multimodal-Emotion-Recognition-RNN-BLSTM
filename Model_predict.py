##################################################################
#
#This function implements a prediction routine to test the previously
#trained NN. This function works for the single audio or text NN.
#
##################################################################


import numpy as np
import os
import csv
import operator
from keras.layers.merge import dot
from keras.models import Model
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
np.seterr(divide='ignore', invalid='ignore')

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
#Main roots
mainRootModelFile = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
mainRoot = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Test')
dirRes = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Z_Results\Recent_Results')
#Features directory paths
dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
dirText = os.path.join(mainRoot + '\FeaturesText')
#Model paths
mainRootModelAudio = os.path.normpath(dirRes + '\RNN_Model_AUDIO_saved.h5')
mainRootModelText = os.path.normpath(dirRes + '\RNN_Model_TEXT_saved.h5')
OutputWeightsPathAudio = os.path.join(dirRes, 'weightsA-improvement-27-0.64.hdf5')
OutputWeightsPathText = os.path.join(dirRes, 'weightsT-improvement-71-0.65.hdf5') 

# --------------------------------------------------------------------------- #
# DEFINE PARAMETERS
# --------------------------------------------------------------------------- #
modelType = 1 #0=OnlyAudio, 1=OnlyText
flagLoadModelAudio = 1 #0=model, 1=weight
flagLoadModelText = 1 #0=model, 1=weight
labelLimit = 170 #Number of each emotion label file to process
nameFileResult = 'PredM_-'+str(modelType)+'-'+'Label_'+str(labelLimit)

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm):
    
    plt.figure(figsize=(4,7))
    
    #NOT NORMALIZED
    print('Confusion matrix, without normalization')
    print(cm)
    plt.subplot(2, 1, 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    #NORMALIZED
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)
    plt.subplot(2, 1, 2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return plt


def computeConfMatrix(allPredictionClasses, expected, dirRes, nameFileResult, flagPlotGraph):
    labelLimit = 170
    expected = np.argmax(expected, axis=1)
    cm = confusion_matrix(expected, allPredictionClasses)
    accurancy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3])/(labelLimit*4)
    print('Accurancy: ',accurancy)
    plt = plot_confusion_matrix(cm, ['joy','ang','sad','neu'], title=nameFileResult+'-CM')
    
    OutputImgPath = os.path.join(dirRes, nameFileResult+'-Acc_'+str(accurancy)+'-CM.png')
    plt.savefig(OutputImgPath)
    if flagPlotGraph:
        plt.show()


def readFeatures(DirRoot):
    listA = [ item for item in os.listdir(DirRoot) if os.path.isfile(os.path.join(DirRoot, item)) ]
    allFileFeature = []
    allFileName = []
    
    i = 0
    #READ encoded audio Features
    for file in listA:
        allFileName.append(file)
        datareader = csv.reader(open(os.path.join(DirRoot, file), 'r'))
        data = []
        for row in datareader:
            data.append([float(val) for val in row])
        Y = np.array([np.array(xi) for xi in data])
        
        #Append all files feature in an unique array
        allFileFeature.append(Y)
        
        if i > labelLimit-1:
            break
        else:
            i += 1
        
    allFileFeature = np.asarray(allFileFeature)
    
    return allFileFeature, allFileName


def organizeFeatures():

    joyAudioFeature, joyFileName = readFeatures(os.path.join(dirAudio, 'joy'))
    angAudioFeature, angFileName = readFeatures(os.path.join(dirAudio, 'ang'))
    sadAudioFeature, sadFileName = readFeatures(os.path.join(dirAudio, 'sad'))
    neuAudioFeature, neuFileName = readFeatures(os.path.join(dirAudio, 'neu'))
    joyTextFeature, joyFileName = readFeatures(os.path.join(dirText, 'joy'))
    angTextFeature, angFileName = readFeatures(os.path.join(dirText, 'ang'))
    sadTextFeature, sadFileName = readFeatures(os.path.join(dirText, 'sad'))
    neuTextFeature, neuFileName = readFeatures(os.path.join(dirText, 'neu'))
    
    #BUILD SHUFFLED FEATURE FILES FOR TRAINING
    allAudioFeature = []
    allTextFeature = []
    allFileName = []
    allLabels = []
    i = 0
    while i < labelLimit:
        if i < len(joyAudioFeature):
            allAudioFeature.append(joyAudioFeature[i])
            allTextFeature.append(joyTextFeature[i])
            allFileName.append(joyFileName[i])
            allLabels.append([[1,0,0,0]])
        
        if i < len(angAudioFeature):
            allAudioFeature.append(angAudioFeature[i])
            allTextFeature.append(angTextFeature[i])
            allFileName.append(angFileName[i])
            allLabels.append([[0,1,0,0]])
        
        if i < len(sadAudioFeature):
            allAudioFeature.append(sadAudioFeature[i])
            allTextFeature.append(sadTextFeature[i])
            allFileName.append(sadFileName[i])
            allLabels.append([[0,0,1,0]])
        
        if i < len(neuAudioFeature):
            allAudioFeature.append(neuAudioFeature[i])
            allTextFeature.append(neuTextFeature[i])
            allFileName.append(neuFileName[i])
            allLabels.append([[0,0,0,1]])

        i +=1
    '''print(np.asarray(allLabels).shape)'''   
    return allAudioFeature, allTextFeature, allFileName, allLabels


def buildBLTSM(maxTimestep, numFeatures):
    
    #SET PARAMETERS
    nb_lstm_cells = 128
    nb_classes = 4
    nb_hidden_units = 512
    
    #MODEL WITH ATTENTION
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)
    input_feature = Input(shape=(maxTimestep,numFeatures))
    x = Masking(mask_value=0.)(input_feature)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)
    alpha = dot([u, y], axes=-1)
    alpha = Activation('softmax')(alpha)
    z = dot([alpha, y], axes=1)
    output = Dense(nb_classes, activation='softmax')(z)
    model = Model(inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['categorical_accuracy'])
    
    return model


def reshapeLSTMInOut(Feat, label, maxTimestep):
    X = []
    X = np.asarray(Feat)
    X = pad_sequences(X, maxlen=maxTimestep, dtype='float32')
    Y = np.asarray(label)
    Y = Y.reshape(len(Y), 4)
    
    return X, Y
 
 
def predictFromModel(model, inputTest, Labels, maxTimestep):
    
    allPrediction = []
    allPredictionClasses = []
    allPredictionClassesMerged = []
    expected = []
    
    #FORMAT X & Y
    X, Y = reshapeLSTMInOut(inputTest, Labels, maxTimestep)
    
    #PREPARE ATTENTION ARRAY INPUT:  training and test
    nb_attention_param = 256
    attention_init_value = 1.0 / 256
    u_train = np.full((X.shape[0], nb_attention_param), attention_init_value, dtype=np.float64)
    
    #PREDICT
    yhat = model.predict([u_train,X])
    for i in range(len(yhat)):
        Pindex, Pvalue = max(enumerate(yhat[i]), key=operator.itemgetter(1))
        allPredictionClasses.append(Pindex)
        allPrediction.append(yhat[i])
        expected.append(Y[i])
    
    #EVALUATE THE MODEL  
    scores = model.evaluate([u_train, X], Y, verbose=0)  
    print('NORMAL EVALUATION %s: %.2f%%' % (model.metrics_names[1], scores[1]*100))         
    
    return allPredictionClasses, allPrediction, expected, yhat
    
    
if __name__ == '__main__':
    
    #EXTRACT FEATURES, NAMES, LABELS, AND ORGANIZE THEM IN AN ARRAY
    allAudioFeature, allTextFeature, allFileName, allLabels = organizeFeatures()
    
    #FIND MAX TIMESTEP FOR PADDING
    maxTimestepAudio = 290 #setted with training because no test file is longer than 290
    maxTimestepText = 85 #text
    
    #MODEL SUMMARY
    print('AUDIO Files with #features: ', allAudioFeature[0].shape[1])
    print('AUDIO Max time step: ',maxTimestepAudio)
    print('TEXT Files with #features: ', allTextFeature[0].shape[1])
    print('TEXT Max time step: ',maxTimestepText)
    print('Predict number of each emotion: ', labelLimit)
     
    #LOAD MODEL OR WIEGHTS FOR AUDIO AND TEXT MODEL
    #Audio
    if modelType == 0:
        if flagLoadModelAudio == 0:
            model_Audio = load_model(mainRootModelAudio) 
        else:    
            model_Audio = buildBLTSM(maxTimestepAudio, allAudioFeature[0].shape[1])
            model_Audio.load_weights(OutputWeightsPathAudio)
    #Text
    if modelType == 1:
        if flagLoadModelText == 0:
            model_Text = load_model(mainRootModelText)   
        else:
            model_Text = buildBLTSM(maxTimestepText, allTextFeature[0].shape[1])
            model_Text.load_weights(OutputWeightsPathText) 
        
    #PREDICT 
    #Audio
    if modelType == 0:
        allPredictionClasses, allPrediction, expected, yhat = predictFromModel(model_Audio, allAudioFeature, allLabels, maxTimestepAudio)
    #Text
    if modelType == 1:
        allPredictionClasses, allPrediction, expected, yhat = predictFromModel(model_Text, allTextFeature, allLabels, maxTimestepText)
        
    #PREDICT & SAVE
    computeConfMatrix(allPredictionClasses, expected, dirRes, nameFileResult, True)
    
    print('END')
    
    