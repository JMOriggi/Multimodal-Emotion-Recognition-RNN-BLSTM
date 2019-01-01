##################################################################
#
#This function implements the training of the NN model based on
#audio and text combined.
#
##################################################################

import operator
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import csv
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Merge
from keras.layers.merge import dot
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.seterr(divide='ignore', invalid='ignore')
import ModelFull_predict as MP

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
#Main roots
mainRoot = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_All')
dirRes = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Z_Results\KFOLD')
#Features directory path
dirAudio = os.path.join(mainRoot + '\FeaturesAudio')
dirText = os.path.join(mainRoot + '\FeaturesText')
#Saved model after training
modelPath = os.path.normpath(dirRes + '\RNN_Model_FULL_saved.h5')
out_file_path = os.path.normpath(dirRes + '\Kfold_Results.txt')
try:
    os.remove(out_file_path)
except OSError:
    pass

# --------------------------------------------------------------------------- #
# DEFINE PARAMETERS
# --------------------------------------------------------------------------- #
flagModel = "Solo" #Full #Solo
nb_lstm_cells = 128 #64 #128
nb_classes = 4
nb_hidden_units = 256 #128 #512
n_epoch = 35 #number of epoch 
batchSize= 20
LRate = 0.001 #0,0001
#BALANCE LABELS AND WEIGHTS: coef1xNumb1=coef2xNumb2
labelLimit = 1700
classWeightFlag = True #Se true usa le info sotto per bilanciare le label.
joyLimit = 1040
angLimit = 1100
sadLimit = 1080
neuLimit = 1700
coef1 = labelLimit/joyLimit
coef2 = labelLimit/angLimit
coef3 = labelLimit/sadLimit
coef4 = labelLimit/neuLimit
class_weight = {0: coef1, 1: coef2, 2: coef3, 3: coef4}
numb_kfold = 5
seed = 7
np.random.seed(seed)

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
    #print(np.asarray(X).shape)
    #print(np.asarray(Y).shape)
    
    return X, Y


def buildBLTSM(numFeaturesAudio, numFeaturesText):
    #MODEL AUDIO WITH ATTENTION
    #Input attention
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)
    #Input Audio and Text
    input_featureAudio = Input(shape=(None, numFeaturesAudio))
    input_featureText = Input(shape=(None, numFeaturesText))
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
    mrg = Merge(mode='concat')([z1,z2]) #mrg = Concatenate([z1,z2])
    #Dense layer and final output
    refOut = Dense(nb_hidden_units, activation='relu')(mrg)
    output = Dense(nb_classes, activation='softmax')(refOut)
    model = Model(inputs=[input_attention, input_featureAudio, input_featureText], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=LRate, rho=0.9, epsilon=None, decay=0.0), metrics=['categorical_accuracy']) #mean_squared_error #categorical_crossentropy
 
    return model


def buildBLTSMSolo(numFeatures):
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)
    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    input_feature = Input(shape=(None,numFeatures))
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


def plotTrainHistory(nameFileResult, history, scores): 
    plt.figure(figsize=(7,10))
    # summarize history for acc and val_acc    
    plt.subplot(2, 1, 1)
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Model categorical accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train set'], loc='upper left')
    # summarize history for loss and val_loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train set'], loc='upper left')
    #save it
    OutputImgPath = os.path.join(dirRes, nameFileResult+'-Train_History-EvAcc_'+str(scores[1]*100)+'.png')
    plt.savefig(OutputImgPath)
    
    
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
    model = buildBLTSM(allAudioFeature[0].shape[1], allTextFeature[0].shape[1])
    SummaryText = 'Att_Model_FULL-RMS-LR_'+str(LRate)+'-BatchSize_'+str(batchSize)+'-FeatNumb_'+str(allAudioFeature[0].shape[1])+'-labelLimit_'+str(labelLimit)
    #MODEL SUMMARY
    model.summary()
    print(SummaryText)
    #KFOLD PARAMS
    kfold = StratifiedKFold(n_splits=numb_kfold, shuffle=True, random_state=seed)
    #PREPARE FEATURES
    train_Audio, train_Y = reshapeLSTMInOut(allAudioFeature, allLabels, maxTimestepAudio)
    train_Text, train_Y = reshapeLSTMInOut(allTextFeature, allLabels, maxTimestepText)
    nb_attention_param = nb_lstm_cells * 2
    attention_init_value = 1.0 / nb_attention_param
    u_train = np.full((train_Audio.shape[0], nb_attention_param), attention_init_value, dtype=np.float64)
    #LAUNCH KFOLD VALIDATION
    outputfile = open(out_file_path, 'a')
    cvscores = []
    foldIndex = 1
    for train_index, test_index in kfold.split(np.zeros(train_Audio.shape[0]),np.zeros(train_Audio.shape[0])):
        print('\n'+"FOLD: ", foldIndex,"/",numb_kfold)
        #Build Model
        if flagModel == "Solo":
            model = buildBLTSMSolo(allAudioFeature[0].shape[1])
        else:
            model = buildBLTSM(allAudioFeature[0].shape[1], allTextFeature[0].shape[1])  
        #Prepare inputs test and train
        X_A = u_train[train_index]
        X_B = train_Audio[train_index]
        X_C = train_Text[train_index]
        Y = train_Y[train_index]
        X_A_2 = u_train[test_index]
        X_B_2 = train_Audio[test_index]
        X_C_2 = train_Text[test_index]
        Y_2 = train_Y[test_index]
        #Train, evaluate and predict current fold
        if flagModel == "Solo":
            if classWeightFlag :
                history = model.fit([X_A, X_B], Y, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2, class_weight=class_weight)  
            else:
                history = model.fit([X_A, X_B], Y, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2)  
            scores = model.evaluate([X_A_2, X_B_2], Y_2, verbose=0)
            yhat = model.predict([X_A_2, X_B_2])
        else:
            if classWeightFlag :
                history = model.fit([X_A, X_B, X_C], Y, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2, class_weight=class_weight)  
            else:
                history = model.fit([X_A, X_B, X_C], Y, batch_size=batchSize, epochs=n_epoch, shuffle=True, verbose=2)  
            scores = model.evaluate([X_A_2, X_B_2, X_C_2], Y_2, verbose=0)
            yhat = model.predict([X_A_2, X_B_2, X_C_2])
        #Print and save results
        allPredictionClasses = []
        allPrediction = []
        expected = []
        for i in range(len(yhat)):
            Pindex, Pvalue = max(enumerate(yhat[i]), key=operator.itemgetter(1))
            allPredictionClasses.append(Pindex)
            allPrediction.append(yhat[i])
            expected.append(Y_2[i])
        #PLOT & SAVE
        nameFileResult = "KFOLD_" + str(foldIndex)
        MP.computeConfMatrix(allPredictionClasses, expected, dirRes, nameFileResult, False)
        plotTrainHistory(nameFileResult, history, scores)
        cvscores.append(scores[1] * 100)
        outputfile.writelines("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)+'\n')
        #Next
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        foldIndex=foldIndex+1
    #Write Summary    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    outputfile.writelines("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))+'\n')
    summary_kfold = "Epoch = "+str(n_epoch)+"  Batch = "+str(batchSize)+"  Labels = "+str(labelLimit)+"  LRate = "+str(LRate)+"  LSTM_Cells = "+str(nb_lstm_cells)+"  Dense_Cells = "+str(nb_hidden_units)+" classWeightFlag = "+str(classWeightFlag)+'\n'
    outputfile.writelines(summary_kfold)
    outputfile.close()
    
    print('END')
    
    