# -*- coding: utf-8 -*-
import gensim
import gensim.models
import os
import sys
myclasses = str(sys.argv[1])
mywindow= int(sys.argv[2])
mysize= int(sys.argv[3])
mincount=int(sys.argv[4])
model =str (sys.argv[5])
pretrain=str (sys.argv[6])
ontology_corpus = str (sys.argv[7])
outfile=str(sys.argv[8])

#### Without pretraining you can define the vector size #######
sentences =gensim.models.word2vec.LineSentence(ontology_corpus)
mymodel =gensim.models.Word2Vec(sentences,sg=1, min_count=mincount, size=300, window=mywindow, sample=1e-3)

word_vectors=mymodel.wv
file= open (outfile, 'w')
with open(myclasses) as f:
	for line in f:
		myclass1=line.rstrip()
		if myclass1 in word_vectors.vocab:
			file.write (str(myclass1) + ' '+ str(mymodel[myclass1]) +'\n')
	file.close()

