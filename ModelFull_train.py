import numpy as np
import os
import csv
import operator
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Embedding, Merge
from keras.layers.merge import dot
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import AveragePooling1D
from keras.layers import Flatten
from keras.layers import Masking
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import itertools
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.seterr(divide='ignore', invalid='ignore')



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
    expected = np.argmax(expected, axis=1)
    cm = confusion_matrix(expected, allPredictionClasses)
    plt = plot_confusion_matrix(cm, ['joy','ang','sad','neu'], title=nameFileResult+'-CM')
    
    OutputImgPath = os.path.join(dirRes, nameFileResult+'_CM.png')
    plt.savefig(OutputImgPath)
    if flagPlotGraph:
        plt.show()



def saveCsv(currentFile, csvOutputFilePath):
    csvOutputFilePath = os.path.join(csvOutputFilePath + '.csv')
    try:
        os.remove(csvOutputFilePath)
    except OSError:
        pass
    
    with open(csvOutputFilePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.asarray(currentFile))
    f.close() 

    
def saveTxt(currentFile, txtOutputFilePath): 
    txtOutputFilePath = os.path.join(txtOutputFilePath + '.txt') 
    try:
        os.remove(txtOutputFilePath)
    except OSError:
        pass
    
    with open(txtOutputFilePath, 'w') as file:      
        for item in currentFile:
            file.write(str(item)+'\n')  
    file.close() 


def readFeatures(DirRoot, labelLimit):
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


def organizeFeaturesV2(dirAudio, dirText, dirLabel, labelLimit):

    joyAudioFeature, joyFileName = readFeatures(os.path.join(dirAudio, 'joy'), labelLimit)
    angAudioFeature, angFileName = readFeatures(os.path.join(dirAudio, 'ang'), labelLimit)
    sadAudioFeature, sadFileName = readFeatures(os.path.join(dirAudio, 'sad'), labelLimit)
    neuAudioFeature, neuFileName = readFeatures(os.path.join(dirAudio, 'neu'), labelLimit)
    joyTextFeature, joyFileName = readFeatures(os.path.join(dirText, 'joy'), labelLimit)
    angTextFeature, angFileName = readFeatures(os.path.join(dirText, 'ang'), labelLimit)
    sadTextFeature, sadFileName = readFeatures(os.path.join(dirText, 'sad'), labelLimit)
    neuTextFeature, neuFileName = readFeatures(os.path.join(dirText, 'neu'), labelLimit)
    joyLabels, joyFileName = readFeatures(os.path.join(dirLabel, 'joy'), labelLimit)
    angLabels, angFileName = readFeatures(os.path.join(dirLabel, 'ang'), labelLimit)
    sadLabels, sadFileName = readFeatures(os.path.join(dirLabel, 'sad'), labelLimit)
    neuLabels, neuFileName = readFeatures(os.path.join(dirLabel, 'neu'), labelLimit)
    '''print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)'''
    
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
            allLabels.append(joyLabels[i])
        
        if i < len(angAudioFeature):
            allAudioFeature.append(angAudioFeature[i])
            allTextFeature.append(angTextFeature[i])
            allFileName.append(angFileName[i])
            allLabels.append(angLabels[i])
        
        if i < len(sadAudioFeature):
            allAudioFeature.append(sadAudioFeature[i])
            allTextFeature.append(sadTextFeature[i])
            allFileName.append(sadFileName[i])
            allLabels.append(sadLabels[i])
        
        if i < len(neuAudioFeature):
            allAudioFeature.append(neuAudioFeature[i])
            allTextFeature.append(neuTextFeature[i])
            allFileName.append(neuFileName[i])
            allLabels.append(neuLabels[i])

        i +=1

    return allAudioFeature, allTextFeature, allFileName, allLabels


def organizeFeatures(dirAudio, dirText, dirLabel, labelLimit):

    joyAudioFeature, joyFileName = readFeatures(os.path.join(dirAudio, 'joy'), labelLimit)
    angAudioFeature, angFileName = readFeatures(os.path.join(dirAudio, 'ang'), labelLimit)
    sadAudioFeature, sadFileName = readFeatures(os.path.join(dirAudio, 'sad'), labelLimit)
    neuAudioFeature, neuFileName = readFeatures(os.path.join(dirAudio, 'neu'), labelLimit)
    joyTextFeature, joyFileName = readFeatures(os.path.join(dirText, 'joy'), labelLimit)
    angTextFeature, angFileName = readFeatures(os.path.join(dirText, 'ang'), labelLimit)
    sadTextFeature, sadFileName = readFeatures(os.path.join(dirText, 'sad'), labelLimit)
    neuTextFeature, neuFileName = readFeatures(os.path.join(dirText, 'neu'), labelLimit)
    joyLabels, joyFileName = readFeatures(os.path.join(dirLabel, 'joy'), labelLimit)
    angLabels, angFileName = readFeatures(os.path.join(dirLabel, 'ang'), labelLimit)
    sadLabels, sadFileName = readFeatures(os.path.join(dirLabel, 'sad'), labelLimit)
    neuLabels, neuFileName = readFeatures(os.path.join(dirLabel, 'neu'), labelLimit)
    '''print(allAudioFeature.shape)
    print(allTextFeature.shape)
    print(allLabels.shape)'''
    
    #BUILD SHUFFLED FEATURE FILES FOR TRAINING
    allAudioFeature = []
    allTextFeature = []
    allFileName = []
    allLabels = []
    i = 0
    while i < labelLimit:
        allAudioFeature.append(joyAudioFeature[i])
        allAudioFeature.append(angAudioFeature[i])
        allAudioFeature.append(sadAudioFeature[i])
        allAudioFeature.append(neuAudioFeature[i])
        
        allTextFeature.append(joyTextFeature[i])
        allTextFeature.append(angTextFeature[i])
        allTextFeature.append(sadTextFeature[i])
        allTextFeature.append(neuTextFeature[i])
        
        allFileName.append(joyFileName[i])
        allFileName.append(angFileName[i])
        allFileName.append(sadFileName[i])
        allFileName.append(neuFileName[i])
        
        allLabels.append(joyLabels[i])
        allLabels.append(angLabels[i])
        allLabels.append(sadLabels[i])
        allLabels.append(neuLabels[i])
        
        i +=1

    return allAudioFeature, allTextFeature, allFileName, allLabels


def reshapeLSTMInOut(audFeat, label, maxTimestep):
    X = []
    X = np.asarray(audFeat)
    X = pad_sequences(X, maxlen=maxTimestep, dtype='float32')
    Y = np.asarray(label)
    Y = Y.reshape(len(Y), 4)
    
    return X, Y


def buildBLTSM(maxTimestepAudio, numFeaturesAudio, maxTimestepText, numFeaturesText, LRate):
    
    nb_lstm_cells = 128
    nb_classes = 4
    nb_hidden_units = 512 #128
        
    #MODEL AUDIO WITH ATTENTION
    #Input attention
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)
    #Input Audio and Text
    input_featureAudio = Input(shape=(maxTimestepAudio, numFeaturesAudio))
    input_featureText = Input(shape=(maxTimestepText, numFeaturesText))
    #Both model parallel structure
    x1 = Masking(mask_value=0.)(input_featureText)
    x1 = Dense(nb_hidden_units, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(nb_hidden_units, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    y1 = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x1)
    x2 = Masking(mask_value=0.)(input_featureAudio)
    x2 = Dense(nb_hidden_units, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(nb_hidden_units, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    y2 = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x2)
    #Attention step parallel for both model
    alpha1 = dot([u, y1], axes=-1)  # inner prod.
    alpha1 = Activation('softmax')(alpha1)
    alpha2 = dot([u, y2], axes=-1)  # inner prod.
    alpha2 = Activation('softmax')(alpha2)
    z1 = dot([alpha1, y1], axes=1)
    z2 = dot([alpha2, y2], axes=1)
    #Merge step
    mrg = Merge(mode='concat')([z1,z2])
    #Dense layer and final output
    refOut = Dense(nb_hidden_units, activation='relu')(mrg)
    output = Dense(nb_classes, activation='softmax')(refOut)
    
    model = Model(inputs=[input_attention, input_featureAudio, input_featureText], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=LRate, rho=0.9, epsilon=None, decay=0.0), metrics=['categorical_accuracy']) #mean_squared_error #categorical_crossentropy

    
    return model


def trainBLSTM(model, allAudioFeature, allTextFeature, Labels, n_epoch, dirRes, maxTimestepAudio, maxTimestepText, batchSize):    
    
    #RESHAPE TRAIN DATA
    train_Audio, train_Y = reshapeLSTMInOut(allAudioFeature, Labels, maxTimestepAudio)
    train_Text, train_Y = reshapeLSTMInOut(allTextFeature, Labels, maxTimestepText)
    
    #CHECPOINT
    #OutputWeightsPath = os.path.join(dirRes, 'weights.best.hdf5')
    OutputWeightsPath = os.path.join(dirRes, 'weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5')
    try:
        os.remove(OutputWeightsPath)
    except OSError:
        pass
    callbacks_list = [
        #EarlyStopping(monitor='val_loss', patience=Patience, verbose=1, mode='auto'),
        ModelCheckpoint(filepath=OutputWeightsPath, monitor='val_categorical_accuracy', save_best_only='True', verbose=1, mode='max')
    ]
    
    #PREPARE ATTENTION ARRAY INPUT:  training and test
    nb_attention_param = 256
    attention_init_value = 1.0 / 256
    u_train = np.full((train_Audio.shape[0], nb_attention_param), attention_init_value, dtype=np.float64)
    
    #FIT MODEL for one epoch on this sequence
    history = model.fit([u_train, train_Audio, train_Text], train_Y, validation_split=0.20, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2, callbacks=callbacks_list)  
        
    #EVALUATION OF THE BEST VERSION MODEL
    modelEv = model
    scores = modelEv.evaluate([u_train, train_Audio, train_Text], train_Y, verbose=0)  
    print('Evaluation model saved %s: %.2f%%' % (modelEv.metrics_names[1], scores[1]*100)) 
        
    return model, history, scores[1]*100  

    
if __name__ == '__main__':
    
    #DEFINE MAIN ROOT
    Computer = 'new'
    #Computer = 'old'
    if Computer == 'new':
        mainRoot = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
        dirRes = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Z_Results\Recent_Results')
    if Computer == 'old':        
        mainRoot = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Corpus_Training')
        dirRes = os.path.normpath(r'D:\DATA\POLIMI\----TESI-----\Z_Results\Recent_Results')
    
    #BUILD PATH FOR EACH FEATURE DIR
    dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
    dirText = os.path.join(mainRoot + '\FeaturesText')
    dirLabel = os.path.join(mainRoot + '\LablesEmotion')
    
    #DEFINE PARAMETERS
    labelLimit = 1300 #720 for balanced, 1300 for max [joy 742, ang 933, sad 839, neu 1324]
    n_epoch = 80 #number of epoch 
    batchSize= 20
    LRateAudio = 0.0001
    
    #EXTRACT FEATURES, NAMES, LABELS, AND ORGANIZE THEM IN AN ARRAY
    allAudioFeature, allTextFeature, allFileName, allLabels = organizeFeaturesV2(dirAudio, dirText, dirLabel, labelLimit)
    
    #FIND MAX TIMESTEP FOR PADDING AUDIO
    maxTimestepAudio = 0 #500
    for z in allAudioFeature:
        zStep = np.asarray(z).shape[0]
        if maxTimestepAudio < zStep:
            maxTimestepAudio = zStep
    
    #FIND MAX TIMESTEP FOR PADDING TEXT
    maxTimestepText = 0
    for z in allTextFeature:
        zStep = np.asarray(z).shape[0]
        if maxTimestepText < zStep:
            maxTimestepText = zStep        
            
    #BUILD MODEL
    model = buildBLTSM(maxTimestepAudio, allAudioFeature[0].shape[1], maxTimestepText, allTextFeature[0].shape[1], LRateAudio)
    SummaryText = 'Att_Model_FULL-RMS-LR_'+str(LRateAudio)+'-BatchSize_'+str(batchSize)+'-FeatNumb_'+str(allAudioFeature[0].shape[1])+'-labelLimit_'+str(labelLimit)
    
    #MODEL SUMMARY
    model.summary()
    print(SummaryText)
    print('Max time step Audio: ',maxTimestepAudio)
    print('Max time step Text: ',maxTimestepText)
    print('Train number of each emotion: ', labelLimit)
    print('Train of #file: ', labelLimit*4)
    
    #TRAIN & SAVE LSTM: considering one at time
    model, history, evAcc = trainBLSTM(model, allAudioFeature, allTextFeature, allLabels, n_epoch, dirRes, maxTimestepAudio, maxTimestepText, batchSize)   
    modelPath = os.path.normpath(dirRes + '\RNN_Model_FULL_saved.h5')
    model.save(modelPath, overwrite=True)
    
    #VISUALIZE HISTORY
    # summarize history for accuracy
    plt.figure(figsize=(5,8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model categorical_accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #save it
    OutputImgPath = os.path.join(dirRes, 'Train_History-EvAcc_'+str(evAcc)+'.png')
    plt.savefig(OutputImgPath)
    plt.show()
    
    print(SummaryText)
    print('END')
    
    