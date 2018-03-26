import numpy as np

a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
print(a)
print(b)
str1 = str(a[0])
str2 = ''.join(str(e) for e in b)
print(str1)
print(str2)
audioname = 'NameOfTheFile'
r=[]
r=audioname+',EXP:'+str1+',AUD:'+str2+',TEXT:'+str2+'\n'
print(r)

file = open('TEst.txt','w')
file.writelines(r)
file.writelines(r)
file.close()

'''import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.moveCopyAudioFiles(mainRoot)'''