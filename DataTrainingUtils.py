#STRUTTURA INIZIALE DA MANTENERE: 
#Corpus[...]
#-> AllAudio, Session1,2,3,4,5[...]
#-->EmoEvaluation[files], Sentences_audio[dirs[wav]], Transcriptions[files]

from keras.preprocessing.text import Tokenizer
import os
import csv
import shutil
import numpy as np


#BUILD A TXT FILE WITH ALL THE USEFULL DATA: <audioFileName>;<emo>;<transcriptionText>
def clusterData(mainRoot):
    print('****Start of method buildDictionary')
    
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) and item[0] == 'S']
    print('All Sessions',dirlist)
    
    #CREATE OUTPUT DATA FILE: remove if it already exist and recreate it new
    outputfilePath = os.path.join(mainRoot+'\AllData.txt')
    try:
        os.remove(outputfilePath)
    except OSError:
        pass
    outputfile = open(outputfilePath, 'a')
    
    #The output file is placed in the root folder and the content will have the format: AudioFileName, CatOutput, ValOutput.
    for session in dirlist:
        print('Parsing: ',session)
        
        #COMPOSE DIRECTORY PATH FOR THE EMOTION RESULTS FILE FOR THE CURRENT SESSION
        directoryEmoPath = os.path.normpath(os.path.join(mainRoot, session)+'\EmoEvaluation')
        emolist = [ item for item in os.listdir(directoryEmoPath) if os.path.isfile(os.path.join(directoryEmoPath, item)) ]
        print('Directory Emotion: ',directoryEmoPath)
        
        #COMPOSE DIRECTORY PATH FOR THE SENTENCE TRANSCRIPTION FILE FOR THE CURRENT SESSION
        directoryText = os.path.normpath(os.path.join(mainRoot, session)+'\Transcriptions')
        translist = [ item for item in os.listdir(directoryText) if os.path.isfile(os.path.join(directoryText, item)) ]
        print('Directory Transcription: ',directoryText)
        
        #PARSE ALL THE FILES AND APPEND IN THE OUTPUT FILE
        for file in emolist:
            with open(os.path.join(directoryEmoPath, file), 'r') as inputfile:
                for lines in inputfile:
                    lines = lines.strip()
                    pos = lines.find('Ses')
                    if pos != -1:
                        #CREATE NEW LINE FOR EMOTION RESULTS
                        audioName = lines.split()[3]
                        emoLabel = lines.split()[4]
                        '''parselines = lines.split()[3]+';'+lines.split()[4]+';'+lines.split()[5]+lines.split()[6]+lines.split()[7]'''
                        #FOR EACH LINE FIND THE CORRESPONDING TRANSCRIPTION SENTENCE IN THE TRANSCRIPTION FILE
                        for file2 in translist:
                            if file2 == file:
                                with open(os.path.join(directoryText, file2), 'r') as inputfile2:
                                    for lines2 in inputfile2:
                                        if lines2.split(' ')[0] == audioName:
                                            transcription = lines2.split(':')[1]
                                            transcription = transcription.split('\n')[0]
                                            transcription = transcription.split(" ", 1)[1]
                                            break
                        
                        #APPEND IN THE OUTPUT FILE                    
                        outputfile.writelines(audioName+';'+emoLabel+';'+transcription+'\n')
            inputfile.close()
    outputfile.close()  
    
    print('****End of method buildDictionary')
    
    
#BUILD 2 CSV FILE: one for all the sentences encoded, one with the index of the filename
def encoder(mainRoot):    
    print('****Start of method encodeText')
    
    #TAKE FROM HERE ALL DATA TO ENCODE
    AllDataPath = os.path.normpath(mainRoot+'\AllData.txt')
    
    #CREATE A TOKENIZER
    t = Tokenizer() 
    
    #READ THE FILE AND BUILD ARRAYS
    #Audio File Names
    with open(AllDataPath, 'r') as AllDatafile:
        X = [line.strip() for line in AllDatafile] 
        arrayFileName = [line.split(';')[0] for line in X] 
    AllDatafile.close()
    #Emotion Labels
    with open(AllDataPath, 'r') as AllDatafile:
        Y = [line.strip() for line in AllDatafile] 
        arrayEmoLabel = [line.split(';')[1] for line in Y]
    AllDatafile.close()  
    #Transcriptions   
    with open(AllDataPath, 'r') as AllDatafile:
        Z = [line.strip() for line in AllDatafile] 
        arrayText = [line.split(';')[2] for line in Z]
    AllDatafile.close()
     
    #PRINT ARRAYS           
    '''print(arrayFileName[0:10])
    print(arrayEmoLabel[0:10])
    print(arrayText[0:10])'''  
    
    #ENCODE EMOTIONS
    i = 0
    emoEncoded = []
    while i < len(arrayEmoLabel):
        emoLabel = arrayEmoLabel[i]
        if  emoLabel == 'exc' or emoLabel == 'hap': 
            code = [1,0,0,0,0,0,0] #JOY
        if  emoLabel == 'ang' or emoLabel == 'fru':    
            code = [0,1,0,0,0,0,0] #ANG
        if  emoLabel == 'dis': 
            code = [0,0,1,0,0,0,0] 
        if  emoLabel == 'sad': 
            code = [0,0,0,1,0,0,0] 
        if  emoLabel == 'sur': 
            code = [0,0,0,0,1,0,0] 
        if  emoLabel == 'fea': 
            code = [0,0,0,0,0,1,0]  
        if  emoLabel == 'neu': 
            code = [0,0,0,0,0,0,1]
        if  emoLabel == 'other' or emoLabel == 'xxx': 
            code = [0,0,0,0,0,0,2]  #NOT CLASSIFIED
        emoEncoded.append(code)
        i += 1
    print(emoEncoded[0:10])
    
    #ENCODE TEXT
    t.fit_on_texts(arrayText)
    encodeText = t.texts_to_sequences(arrayText)
    print(t.word_index)
    print(encodeText[0:10])
    
    #CREATE OUTPUTS DATA FILE: remove if it already exist and recreate it new
    outputfile1Path = os.path.join(mainRoot+'\AudioFilesName.csv')
    outputfile2Path = os.path.join(mainRoot+'\encodedEmo.csv')
    outputfile3Path = os.path.join(mainRoot+'\encodedText.csv')
    try:
        os.remove(outputfile1Path)
        os.remove(outputfile2Path)
        os.remove(outputfile3Path)
    except OSError:
        pass
    
    #WRITE CSV FILE
    #Audio File Names
    with open(outputfile1Path, 'w', newline='') as csvfile1:
        writer = csv.writer(csvfile1)
        for name in sorted(arrayFileName):
            writer.writerow([name])
    csvfile1.close()
    #Emotion Labels
    with open(outputfile2Path, 'w', newline='') as csvfile2:
        writer = csv.writer(csvfile2)
        writer.writerows(emoEncoded)
    csvfile2.close()
    #Transcriptions 
    with open(outputfile3Path, 'w', newline='') as csvfile3:
        writer = csv.writer(csvfile3)
        writer.writerows(encodeText)
    csvfile3.close()
    
    print('****End of method encodeText')
 
    
#MOVE ALL THE AUDIO FILES IN THE MAINROOT IN 1 DIRECTORY    
def moveCopyAudioFiles(mainRoot):
    print('****Start of method moveAudioFiles')
    
    #SET DESTINATION PATH
    destPath = os.path.normpath(mainRoot+'\AllAudioFiles')
    print('DestPath: ',destPath)
    
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    sessDirList = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) and item[0] == 'S']
    print('All Sessions',sessDirList)
    
    for session in sessDirList:
        currentAudioDirPath = os.path.normpath(os.path.join(mainRoot, session)+'\Sentences_audio')
        audioGroupDir = [ item for item in os.listdir(currentAudioDirPath) if os.path.isdir(os.path.join(currentAudioDirPath, item)) ]
        #destPath = currentAudioDirPath
        print('Inside: ',session)
        
        for audioGroup in audioGroupDir:
            currentAudioGroupPath = os.path.normpath(os.path.join(currentAudioDirPath, audioGroup))
            audlist = [ item for item in os.listdir(currentAudioGroupPath) if os.path.isfile(os.path.join(currentAudioGroupPath, item)) ]
            print('Inside audioGroup: ',audioGroup)
               
            for Afile in audlist:
                print('Moving file: ',Afile)
                audioFilePath = os.path.join(currentAudioGroupPath, Afile)
                shutil.copy(audioFilePath, destPath)
                #shutil.move(audioFilePath, destPath)
                
    print('****End of method moveAudioFiles')    


#READ ALREADY CREATED CSV DATA FOR TRAINING ANG RETURN ALL ARRAYS    
def readCsvData(mainRoot):
    print('****Start of method readCsvData')
    
    #READ Audio File Name 
    datareader = csv.reader(open(os.path.join(mainRoot+'\AudioFilesName.csv'), 'r'))
    data = []
    for row in datareader:
        data.append(row)
    X = data
    
    #READ encoded emotion: Read the content as an array of numbers and not string as default
    datareader = csv.reader(open(os.path.join(mainRoot+'\encodedEmo.csv'), 'r'))
    data = []
    for row in datareader:
        data.append([int(val) for val in row])
    Y = np.array([np.array(xi) for xi in data])
    
    #READ encoded text: Read the content as an array of numbers and not string as default
    datareader = csv.reader(open(os.path.join(mainRoot+'\encodedText.csv'), 'r'))
    data = []
    for row in datareader:
        data.append([int(val) for val in row])
    Z = np.array([np.array(xi) for xi in data])
    
    print('****End of method readCsvData')
    return X, Y, Z    
    
    