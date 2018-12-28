##################################################################
#
#This function aim to reorganize data from the adapted corpus to 
#prepare it for training and text. It create a txt file with in each
#line the audiofilename the corresponding emotion label and the 
#transcription for each sentence. This txt file will be used from
#training and text version to access more easily all the data involved.
#Also this function will move all the audio file in one single folder,
#again to let training and test access files more easily.
#
##################################################################


import os
import shutil
import numpy as np

# --------------------------------------------------------------------------- #
# DEFINE PATHS
# --------------------------------------------------------------------------- #
#Main roots
Computer = 'All'
#Computer = 'test'
if Computer == 'All':
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_All')
if Computer == 'training':
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Training')
if Computer == 'test':    
    main_root = os.path.normpath(r'C:\DATA\POLIMI\----TESI-----\Corpus_Test')
#Directory paths
ZData_path = os.path.join(main_root + '\ZData')
out_file_path =  os.path.join(main_root+'\AllData.txt') 
audio_file_dest_path = os.path.normpath(main_root+'\AllAudioFiles')


#BUILD A TXT FILE WITH ALL THE USEFULL DATA: <audioFileName>;<emo>;<transcriptionText>
def clusterData():
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    dirlist = [ item for item in os.listdir(ZData_path) if os.path.isdir(os.path.join(ZData_path, item))]
    print('All Sessions',dirlist)
    
    #CREATE OUTPUT DATA FILE: remove if it already exist and recreate it new
    try:
        os.remove(out_file_path)
    except OSError:
        pass
    outputfile = open(out_file_path, 'a')
    
    #The output file is placed in the root folder and the content will have the format: AudioFileName, CatOutput, ValOutput.
    for session in dirlist:
        print('Parsing: ',session)
        
        #COMPOSE DIRECTORY PATH: for emotion labels file of the current session
        directoryEmoPath = os.path.normpath(os.path.join(ZData_path, session)+'\EmoEvaluation')
        emolist = [ item for item in os.listdir(directoryEmoPath) if os.path.isfile(os.path.join(directoryEmoPath, item)) ]
        print('Directory Emotion: ',directoryEmoPath)
        
        #COMPOSE DIRECTORY PATH: for the transcription file of current session
        directoryText = os.path.normpath(os.path.join(ZData_path, session)+'\Transcriptions')
        translist = [ item for item in os.listdir(directoryText) if os.path.isfile(os.path.join(directoryText, item)) ]
        print('Directory Transcription: ',directoryText)
        
        #PARSE ALL THE FILES AND APPEND DATA IN THE OUTPUT FILE
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


#MOVE ALL THE AUDIO FILES IN THE main_root IN 1 DIRECTORY    
def moveCopyAudioFiles():
    print('****Start of method moveAudioFiles')
    
    #SET DESTINATION PATH
    print('audio_file_dest_path: ',audio_file_dest_path)
    
    #GET ALL THE SESSIONS DIRECTORY NAME FROM MAIN ROOT
    sessDirList = [ item for item in os.listdir(ZData_path) if os.path.isdir(os.path.join(ZData_path, item))]#[ item for item in os.listdir(main_root) if os.path.isdir(os.path.join(main_root, item)) and item[0] == 'S']
    print('All Sessions',sessDirList)
    
    for session in sessDirList:
        currentAudioDirPath = os.path.normpath(os.path.join(ZData_path, session)+'\Sentences_audio')
        audioGroupDir = [ item for item in os.listdir(currentAudioDirPath) if os.path.isdir(os.path.join(currentAudioDirPath, item)) ]

        print('Inside: ',session)
        
        for audioGroup in audioGroupDir:
            currentAudioGroupPath = os.path.normpath(os.path.join(currentAudioDirPath, audioGroup))
            audlist = [ item for item in os.listdir(currentAudioGroupPath) if os.path.isfile(os.path.join(currentAudioGroupPath, item)) ]
            print('Inside audioGroup: ',audioGroup)
               
            for Afile in audlist:
                print('Copy file: ',Afile)
                audioFilePath = os.path.join(currentAudioGroupPath, Afile)
                shutil.copy(audioFilePath, audio_file_dest_path)
                #shutil.move(audioFilePath, audio_file_dest_path)
                
    print('****End of method moveAudioFiles')   

    
if __name__ == '__main__':
    
    clusterData()  
    moveCopyAudioFiles()    
        
    print('END') 
        