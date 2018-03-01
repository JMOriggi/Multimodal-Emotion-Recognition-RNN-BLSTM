from DataTrainingUtils import DataTrainingUtils


trainData = DataTrainingUtils()
trainData.setDataCorpus()

output, emo, val, text = trainData.getOutputDataFromAudio('Ses04F_script01_1_M019')
print('---Coresponding output for Audio Ses04F_script01_1_M019---')
print('Name: ',output.split(';')[0],'\nEmotion: ',emo,'\nValence: ',val,'\nTranscription: ',text)            
            