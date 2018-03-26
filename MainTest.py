import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
print(a)
print(b)
with open(r'TESTT.txt', 'w') as f:
    f.write(" ".join(map(str, a)))

'''import os
import DataTrainingUtils as trainData
mainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
trainData.moveCopyAudioFiles(mainRoot)'''