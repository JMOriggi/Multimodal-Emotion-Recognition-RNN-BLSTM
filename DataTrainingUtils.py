#Se cambia posizione cartella modificare SOLO la root e mantenere struttura cartelle uguale.
#Struttura da mantenere: Corpus-->Session1,2,3,4,5-->EmoEvaluation,Sentences_audio,Transcriptions

import os

mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus')

class DataTrainingUtils:
    
    def __init__(self):
        print('****Initiate Class TrainingUtils')
    
    
    @staticmethod
    def setDataCorpus():
        print('****Start of method setDataCorpus')
        
        #Get all the sessions directory name from the main root
        dirlist = [ item for item in os.listdir(mainRoot) if os.path.isdir(os.path.join(mainRoot, item)) ]
        #print('All Sessions',dirlist)
        
        #Create a unique txt file for each Session in witch we can find all the training output for each sentence.
        #The output file is placed in the root folder.
        #The output training file content will have the format: SessionTypeName, AudioFileName, CatOutput, ValOutput.
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
     
        
    def getOutputDataFromAudio(self, audioFileName):
        print('****Start of method getOutputFromAudio')
        #print(audioFileName)
        onlyfiles = [f for f in os.listdir(mainRoot) if os.path.isfile(os.path.join(mainRoot, f))]
        for file in onlyfiles:
            if file.split('.')[1] == 'txt':
                #print('Current file: ',file)
                with open(os.path.join(mainRoot, file), 'r') as inputfile:
                    for lines in inputfile:
                        if lines.split(';')[0] == audioFileName:
                            output = lines
                            break
        
        emo = output.split(';')[1]
        val = output.split(';')[2]
        text = output.split(';')[3]                  
        print('****End of method getOutputFromAudio')                
        return output, emo, val, text
      