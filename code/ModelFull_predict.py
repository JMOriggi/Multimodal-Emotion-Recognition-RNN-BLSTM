##################################################################
#
#This function implements a prediction routine to test the previously
#trained NN. This function works for the audio and text combined NN.
#
##################################################################


import numpy as np
import os
import csv
import operator
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
np.seterr(divide='ignore', invalid='ignore')
from ModelFull_train import buildBLTSM as BLSTMFullModel

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
#Mains roots
mainRoot = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Test')
dirRes = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Z_Results\Recent_Results')
#Features paths
dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
dirText = os.path.join(mainRoot + '\FeaturesText')
#Model and weights paths (only one mandatory)
mainRootModel = os.path.join(dirRes, 'RNN_Model_FULL_saved.h5')
OutputWeightsPath = os.path.join(dirRes, 'weights-improvement-99-0.89.hdf5')

# --------------------------------------------------------------------------- #
# DEFINE PARAMETERS
# --------------------------------------------------------------------------- #
flagLoadModel = 1 #0=model, 1=weight
labelLimit = 384 #170 for balanced, 380 for max [joy 299, ang 170, sad 245, neu 384] TOT 1098
allfile = 1098
nameFileResult = 'PredW99-FULL-Label_'+str(allfile)
#Max timestep used for padding, setted according to the training model (NOT MODIFY)
maxTimestepAudio = 290 
maxTimestepText = 85 

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
    
    return plt, cm


def computeConfMatrix(allPredictionClasses, expected, dirRes, nameFileResult, flagPlotGraph):
    
    expected = np.argmax(expected, axis=1)
    cmUA = confusion_matrix(expected, allPredictionClasses)
    plt, cmWA = plot_confusion_matrix(cmUA, ['joy','ang','sad','neu'], title='Confusion Matrix')
    
    accurancyUA = (cmUA[0][0] + cmUA[1][1] + cmUA[2][2] + cmUA[3][3])/(allfile)
    print('Accurancy UA: ',accurancyUA)
    accurancyWA = (cmWA[0][0] + cmWA[1][1] + cmWA[2][2] + cmWA[3][3])/4
    print('Accurancy WA: ',accurancyWA)
    
    accurancy = max(accurancyUA, accurancyWA)
    
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


def reshapeLSTMInOut(Feat, label, maxTimestep):
    X = []
    X = np.asarray(Feat)
    X = pad_sequences(X, maxlen=maxTimestep, dtype='float32')
    Y = np.asarray(label)
    Y = Y.reshape(len(Y), 4)
    
    return X, Y
 
 
def predictFromModel(model, inputTestAudio, inputTestText, Labels, maxTimestepAudio, maxTimestepText):
    
    allPrediction = []
    allPredictionClasses = []
    allPredictionClassesMerged = []
    expected = []
    
    #FORMAT X & Y
    X_Audio, Y = reshapeLSTMInOut(inputTestAudio, Labels, maxTimestepAudio)
    X_Text, Y = reshapeLSTMInOut(inputTestText, Labels, maxTimestepText)
    
    #PREPARE ATTENTION ARRAY INPUT:  training and test
    nb_attention_param = 256
    attention_init_value = 1.0 / 256
    u_train = np.full((X_Audio.shape[0], nb_attention_param), attention_init_value, dtype=np.float64)
    
    #PREDICT
    yhat = model.predict([u_train, X_Audio, X_Text])
    for i in range(len(yhat)):
        #print('Expected:', Y[i], 'Predicted', yhat[i])
        Pindex, Pvalue = max(enumerate(yhat[i]), key=operator.itemgetter(1))
        allPredictionClasses.append(Pindex)
        allPrediction.append(yhat[i])
        expected.append(Y[i])
    
    #EVALUATE THE MODEL  
    scores = model.evaluate([u_train, X_Audio, X_Text], Y, verbose=0)  
    print('NORMAL EVALUATION %s: %.2f%%' % (model.metrics_names[1], scores[1]*100))         
    
    return allPredictionClasses, allPrediction, expected
    
    
if __name__ == '__main__':
    
    #EXTRACT FEATURES, NAMES, LABELS, AND ORGANIZE THEM IN AN ARRAY
    allAudioFeature, allTextFeature, allFileName, allLabels = organizeFeatures()
    
    #MODEL SUMMARY
    print('AUDIO Files with #features: ', allAudioFeature[0].shape[1])
    print('AUDIO Max time step: ',maxTimestepAudio)
    print('TEXT Files with #features: ', allTextFeature[0].shape[1])
    print('TEXT Max time step: ',maxTimestepText)
    print('Predict number of each emotion: ', labelLimit)
     
    #LOAD MODEL OR WIEGHTS
    if flagLoadModel == 0:
        model = load_model(mainRootModel) 
    else:
        model = BLSTMFullModel(allAudioFeature[0].shape[1], allTextFeature[0].shape[1])
        model.load_weights(OutputWeightsPath)
        
    #PREDICT  
    allPredictionClasses, allPrediction, expected = predictFromModel(model, allAudioFeature, allTextFeature, allLabels, maxTimestepAudio, maxTimestepText)
       
    #PREDICT & SAVE
    computeConfMatrix(allPredictionClasses, expected, dirRes, nameFileResult, True)
    
    print('END')
    
    