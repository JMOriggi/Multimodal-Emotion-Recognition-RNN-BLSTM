#Struttura da mantenere: Corpus--> TrainOutput.txt Session1,2,3,4,5-->EmoEvaluation,Sentences_audio,Transcriptions

import os

def setDataCorpus(mainRoot):
    print('****Start of method setDataCorpus')
    
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
    print('All Sessions',dirlist)
    
    #CREATE A UNIQUE TXT FILE FOR EACH SESSION WITH ALL THE TRAINING OUTPUT RESULT FOR EACH AUDIO FILE
    #The output file is placed in the root folder and the content will have the format: AudioFileName, CatOutput, ValOutput.
    for session in dirlist:
        print('Parsing: ',session)
        
        #COMPOSE DIRECTORY PATH FOR THE EMOTION RESULTS FILE FOR THE CURRENT SESSION
        directoryEmoPath = os.path.normpath(os.path.join(mainRoot, session)+'\EmoEvaluation')
        emolist = [ item for item in os.listdir(directoryEmoPath) if os.path.isfile(os.path.join(directoryEmoPath, item)) ]
        print('Directory Emotion: ',directoryEmoPath)
        
        #COMPOSE DIRECTORY PATH FOR THE SENTENCE TRANSCRIPTION FILE FOR THE CURRENT SESSION
        directoryText = os.path.normpath(os.path.join(mainRoot, session)+'\Transcriptions')
        translist = [ item for item in os.listdir(directoryEmoPath) if os.path.isfile(os.path.join(directoryEmoPath, item)) ]
        print('Directory Transcription: ',directoryText)
        
        #CREATE OUTPU DATA FILE: remove if it already exist and recreate it new
        outputfilePath = os.path.join(mainRoot+'\TrainOutput'+session+'.txt')
        try:
            os.remove(outputfilePath)
        except OSError:
            pass
        outputfile = open(outputfilePath, 'a')
        
        #PARSE ALL THE FILES AND APPEND IN THE OUTPUT FILE
        for file in emolist:
            with open(os.path.join(directoryEmoPath, file), 'r') as inputfile:
                for lines in inputfile:
                    lines = lines.strip()
                    pos = lines.find('Ses')
                    if pos != -1:
                        #CREATE NEW LINE FOR EMOTION RESULTS
                        parselines = lines.split()[3]+';'+lines.split()[4]+';'+lines.split()[5]+lines.split()[6]+lines.split()[7]
                        
                        #FOR EACH LINE FIND THE CORRESPONDING TRANSCRIPTION SENTENCE IN THE TRANSCRIPTION FILE
                        for file2 in translist:
                            if file2 == file:
                                with open(os.path.join(directoryText, file2), 'r') as inputfile2:
                                    for lines2 in inputfile2:
                                        if lines2.split(' ')[0] == lines.split()[3]:
                                            transcription = lines2.split(':')[1]
                                            transcription = transcription.split('\n')[0]
                                            #print('Transcription: ',transcription)
                                            break
                        
                        #APPEND IN THE OUTPUT FILE                    
                        outputfile.writelines(parselines+';'+'{'+transcription+'\n')
            inputfile.close()
        outputfile.close()
        
    print('****End of method setDataCorpus\n')
 
    
def getOutputDataFromAudio(audioFileName, mainRoot):
    print('****Start of method getOutputFromAudio')
    
    #TAKE THE OUTPUT TRAINING INFO FROM TRAINING DATA FILE PREVIOUSLY GENERATED
    onlyfiles = [f for f in os.listdir(mainRoot) if os.path.isfile(os.path.join(mainRoot, f))]
    for file in onlyfiles:
        if file.split('.')[1] == 'txt':
            with open(os.path.join(mainRoot, file), 'r') as inputfile:
                for lines in inputfile:
                    if lines.split(';')[0] == audioFileName:
                        output = lines
                        break
    
    #CREATE OUTPUT VARIABLES
    emo = output.split(';')[1]
    val = output.split(';')[2]
    text = output.split('{')[1] 
    code = []
     
    if  emo == 'exc' or emo == 'hap': 
        code = [1,0,0,0,0,0,0] #JOY
    if  emo == 'ang' or emo == 'fru':    
        code = [0,1,0,0,0,0,0] #ANG
    if  emo == 'dis': 
        code = [0,0,1,0,0,0,0] 
    if  emo == 'sad': 
        code = [0,0,0,1,0,0,0] 
    if  emo == 'sur': 
        code = [0,0,0,0,1,0,0] 
    if  emo == 'fea': 
        code = [0,0,0,0,0,1,0]  
    if  emo == 'other' or emo == 'neu' or emo == 'xxx': 
        code = [0,0,0,0,0,0,1]  #NOT CLASSIFIED
                   
    print('****End of method getOutputFromAudio\n')                
    return code, output, emo, val, text
