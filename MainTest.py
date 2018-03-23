from keras.preprocessing.text import Tokenizer
import csv
import numpy as np
# define 5 documents
docs = ['Well done! Well done',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!','klass']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_index)
X = t.texts_to_sequences(docs)
print(X)
Xdata= []
thefile = open('testttt.txt', 'w')
for item in X:
  thefile.write("%s\n" % item)
thefile.close()
with open('testttt.txt', 'r') as f:
    for line in f:
        print(line)
Xdata.append(line)
print(Xdata)




