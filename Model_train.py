##################################################################
#
#This function implements the training of the NN model based on
#audio or text.
#
##################################################################


import numpy as np
import os
import csv
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.seterr(divide='ignore', invalid='ignore')

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
#Main roots
mainRoot = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
dirRes = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Z_Results\Recent_Results')
#Features directory paths
dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
dirText = os.path.join(mainRoot + '\FeaturesText')
#Model paths
mainRootModelAudio = os.path.normpath(mainRoot + '\RNN_Model_AUDIO_saved.h5')
mainRootModelText = os.path.normpath(mainRoot + '\RNN_Model_TEXT_saved.h5')

# --------------------------------------------------------------------------- #
# DEFINE PARAMETERS
# --------------------------------------------------------------------------- #
modelType = 0 #0=Audio, 1=Text
labelLimit = 740 #Number of each emotion label file to process
n_epoch = 200 #number of epoch 
batchSizeAudio = 30
batchSizeText = 20
LRateAudio = 0.001
LRateText = 0.0001
FlagValSet = False #use validation set or not
FlagEarlyStop = False #use earlystop or not (if true set patience epoch for audio and text, and validation set will be considered mandatory)
PatienceAudio = 40
PatienceText = 100
if FlagEarlyStop == True:
    FlagValSet = True

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

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


def reshapeLSTMInOut(audFeat, label, maxTimestep):
    X = np.asarray(audFeat)
    X = pad_sequences(X, maxlen=maxTimestep, dtype='float32')
    Y = np.asarray(label)
    Y = Y.reshape(len(Y), 4)
    
    return X, Y


def buildBLTSM(maxTimestep, numFeatures, LRate):
    
    #MODEL WITH ATTENTION
    nb_lstm_cells = 128
    nb_classes = 4
    nb_hidden_units = 512
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)
    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    input_feature = Input(shape=(maxTimestep,numFeatures))
    x = Masking(mask_value=0.)(input_feature)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)
    # To compute the final weights for the frames which sum to unity
    alpha = dot([u, y], axes=-1)  # inner prod.
    alpha = Activation('softmax')(alpha)
    # Weighted pooling to get the utterance-level representation
    z = dot([alpha, y], axes=1)
    # Get posterior probability for each emotional class
    output = Dense(nb_classes, activation='softmax')(z)
    model = Model(inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=LRate, rho=0.9, epsilon=None, decay=0.0), metrics=['categorical_accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['categorical_accuracy']) 
    
    return model


def trainBLSTM(model, Features, Labels, maxTimestep, batchSize, Patience):    
    
    #RESHAPE TRAIN DATA
    train_X, train_Y = reshapeLSTMInOut(Features, Labels, maxTimestep)
    
    #CHECPOINT
    if FlagEarlyStop:
        OutputWeightsPath = os.path.join(dirRes, 'weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5')
        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=Patience, verbose=1, mode='auto'),
            ModelCheckpoint(filepath=OutputWeightsPath, monitor='val_categorical_accuracy', save_best_only='True', verbose=1, mode='max')
        ]
    else:
        OutputWeightsPath = os.path.join(dirRes, 'weights-improvement-{epoch:02d}-{categorical_accuracy:.2f}.hdf5')    
        callbacks_list = [
            ModelCheckpoint(filepath=OutputWeightsPath, monitor='categorical_accuracy', save_best_only='True', verbose=1, mode='max')
        ]
    
    #PREPARE ATTENTION ARRAY INPUT:  training and test
    nb_attention_param = 256
    attention_init_value = 1.0 / 256
    u_train = np.full((train_X.shape[0], nb_attention_param), attention_init_value, dtype=np.float64)
    
    #FIT MODEL for one epoch on this sequence
    if FlagValSet:
        history = model.fit([u_train, train_X], train_Y, validation_split=0.20, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2, callbacks=callbacks_list)  
    else:
        history = model.fit([u_train, train_X], train_Y, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2, callbacks=callbacks_list)  
        
    #EVALUATION OF THE BEST VERSION MODEL
    modelEv = model
    scores = modelEv.evaluate([u_train, train_X], train_Y, verbose=0)  
    print('Evaluation model saved %s: %.2f%%' % (modelEv.metrics_names[1], scores[1]*100)) 
        
    return model, history, scores[1]*100  

    
if __name__ == '__main__':
    
    #EXTRACT FEATURES, NAMES, LABELS, AND ORGANIZE THEM IN AN ARRAY
    allAudioFeature, allTextFeature, allFileName, allLabels = organizeFeatures()
    
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
    if modelType == 0:
        model = buildBLTSM(maxTimestepAudio, allAudioFeature[0].shape[1], LRateAudio)
        SummaryText = 'Att_Model_'+str(modelType)+'-RMS-LR_'+str(LRateAudio)+'-BatchSize_'+str(batchSizeAudio)+'-FeatNumb_'+str(allAudioFeature[0].shape[1])+'-labelLimit_'+str(labelLimit)
    else:
        model = buildBLTSM(maxTimestepText, allTextFeature[0].shape[1], LRateText)
        SummaryText = 'Att_Model_'+str(modelType)+'-RMS-LR_'+str(LRateText)+'-BatchSize_'+str(batchSizeText)+'-FeatNumb_'+str(allTextFeature[0].shape[1])+'-labelLimit_'+str(labelLimit) 
    
    #MODEL SUMMARY
    model.summary()
    print(SummaryText)
    print('Max time step Audio: ',maxTimestepAudio)
    print('Max time step Text: ',maxTimestepText)
    print('Train number of each emotion: ', labelLimit)
    print('Train of #file: ', labelLimit*4)
    
    #TRAIN & SAVE LSTM: considering one at time
    if modelType == 0:
        model_Audio, history, evAcc = trainBLSTM(model, allAudioFeature, allLabels, n_epoch, dirRes, maxTimestepAudio, batchSizeAudio, PatienceAudio)
        modelPathAudio = os.path.normpath(dirRes + '\RNN_Model_AUDIO_saved.h5')
        model_Audio.save(modelPathAudio, overwrite=True)       
    if modelType == 1:
        model_Text, history, evAcc = trainBLSTM(model, allTextFeature, allLabels, n_epoch, dirRes, maxTimestepText, batchSizeText, PatienceText)    
        modelPathText = os.path.normpath(dirRes + '\RNN_Model_TEXT_saved.h5')
        model_Text.save(modelPathText, overwrite=True)
    
    #VISUALIZE HISTORY
    plt.figure(figsize=(5,8))
    if FlagValSet:
        # summarize history for val_acc
        plt.subplot(2, 1, 1)
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model categorical_accuracy')
        plt.ylabel('categorical_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for val_loss
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
    else:
        # summarize history for acc and val_acc    
        plt.subplot(2, 1, 1)
        plt.plot(history.history['categorical_accuracy'])
        plt.title('model categorical_accuracy')
        plt.ylabel('categorical_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        # summarize history for loss and val_loss
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
    #save it
    OutputImgPath = os.path.join(dirRes, 'Train_History-EvAcc_'+str(evAcc)+'.png')
    plt.savefig(OutputImgPath)
    plt.show()
    
    print(SummaryText)
    print('END')
    
    