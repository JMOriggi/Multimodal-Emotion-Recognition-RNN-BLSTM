#TEST DATA TRAINING CLASS
#from DataTrainingUtils import DataTrainingUtils
#trainData = DataTrainingUtils()
#trainData.setDataCorpus()
#output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
#print('---Coresponding output for Audio Ses04F_script01_1_M019---')
#print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text)            
 
    
 
#TEST AUDIO CLASS     
import AudioUtils as aud
audioFileName ='file.wav'
arrayAudio, sampleRate = aud.getArrayFromAudio(audioFileName)

allFrame = aud.getFrameArray(arrayAudio, sampleRate, 10)

#arrayFFT = aud.getSpectrumFrameArray(arrayAudio)
#print('Returned: ', arrayFFT)

