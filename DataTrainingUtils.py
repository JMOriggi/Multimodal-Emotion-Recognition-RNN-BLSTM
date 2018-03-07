#Se cambia posizione cartella modificare SOLO la root e mantenere struttura cartelle uguale.
#Struttura da mantenere: Corpus-->Session1,2,3,4,5-->EmoEvaluation,Sentences_audio,Transcriptions

import os

#mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus')
mainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus')

def setDataCorpus():
    print('****Start of method setDataCorpus')
    
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
    #print('All Sessions',dirlist)
    
    #Create a unique txt file for each Session in witch we can find all the training output for each sentence.
    #The output file is placed in the root folder.
    #The output training file content will have the format: AudioFileName, CatOutput, ValOutput.
    for session in dirlist:
        print('Parsing: ',session)
        #Create directory path for the emotion results for the current session
        currentSessionPathEmo = os.path.join(mainRoot, session)
        currentSessionPathEmo += '\EmoEvaluation'
        directoryEmo = os.path.normpath(currentSessionPathEmo)
        #print('Current directory Emotion: ',directoryEmo)
        
        #Create directory path for the sentence transcription for the current session
        currentSessionPathText = os.path.join(mainRoot, session)
        currentSessionPathText += '\Transcriptions'
        directoryText = os.path.normpath(currentSessionPathText)
        #print('Current directory Transcription: ',directoryText)
        
        #Parse all the txt file and create the output file
        #outputfile = open(os.path.join(mainRoot, session)+'\TrainOutput'+session+'.txt', 'a')
        #outputfile = open('TrainOutput'+session+'.txt', 'a')
        outputfile = open(mainRoot+'\TrainOutput'+session+'.txt', 'a')
        for dirs, subdir, files in os.walk(directoryEmo):
            #print('All File in: ',files)
            for file in files:
                #print('Open and parsing: ',file.split('.')[0])
                with open(os.path.join(directoryEmo, file), 'r') as inputfile:
                    for lines in inputfile:
                        lines = lines.strip()
                        pos = lines.find('Ses')
                        if pos != -1:
                            parselines = lines.split()[3]+';'+lines.split()[4]+';'+lines.split()[5]+lines.split()[6]+lines.split()[7]
                            #print(lines.split()[3])
                            
                            #for each line find the corresponding transcription sentence
                            for dirs2, subdir2, files2 in os.walk(directoryText):
                                for file2 in files2:
                                    if file2 == file:
                                        with open(os.path.join(directoryText, file2), 'r') as inputfile2:
                                            for lines2 in inputfile2:
                                                if lines2.split(' ')[0] == lines.split()[3]:
                                                    transcription = lines2.split(':')[1]
                                                    transcription = transcription.split('\n')[0]
                                                    #print('Transcription: ',transcription)
                                                    break
                            #outputfile.writelines(file.split('.')[0]+','+parselines+','+'{'+transcription+'}'+'\n')
                            outputfile.writelines(parselines+';'+'{'+transcription+'}'+'\n')
                inputfile.close()
        outputfile.close()
    print('****End of method setDataCorpus')
 
    
def getOutputDataFromAudio(audioFileName):
    print('****Start of method getOutputFromAudio')
    
    #TAKE THE OUTPUT TRAINING INFO FROM TRAINING DATA FILE PREVIOUSLY GENERATED
    onlyfiles = [f for f in os.listdir(mainRoot) if os.path.isfile(os.path.join(mainRoot, f))]
    for file in onlyfiles:
        print('a', file)
        if file.split('.')[1] == 'txt':
            with open(os.path.join(mainRoot, file), 'r') as inputfile:
                for lines in inputfile:
                    if lines.split(';')[0] == audioFileName:
                        output = lines
                        break
    
    #CREATE OUTPUT VARIABLES
    emo = output.split(';')[1]
    val = output.split(';')[2]
    text = output.split(';')[3] 
    code = ''
    if  emo == 'ext': 
        code = 1
    if  emo == 'ang':    
        code = 2
    if  emo == 'fru': 
        code = 3
    if  emo == 'sad': 
        code = 4
    if  emo == 'sur': 
        code = 5 
    if  emo == 'neu': 
        code = 6 
    if  emo == 'xxx': 
        code = 7 
    if  emo == 'other': 
        code = 8 
    if  emo == 'test': 
        code = 9     
                   
    print('****End of method getOutputFromAudio')                
    return code, output, emo, val, text
  