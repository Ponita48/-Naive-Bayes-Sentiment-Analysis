import csv
import re
import string
import nltk
# nltk.download('punkt')
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

def clean_instagram(instagram): 
	''' 
	Utility function to clean tweet text by removing links, special characters 
	using simple regex statements. 

	'''
	return ' '.join(re.sub(r'[^\w\s]','',instagram).split())
	
def stopword(instagram):
	stopword_factory = StopWordRemoverFactory()
	stopword = stopword_factory.create_stop_word_remover()
	return stopword.remove(instagram)

def stemming(instagram):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	return stemmer.stem(instagram)

#''', error_bad_lines=False'''

data = pd.read_csv('DATA TRAINING.csv', sep=';')
captions = data['Text'].values

# captions = ['Maaruf sangat yakin dukungan Mbah Moen akan berdampak signifikan kepada pasangan Jokowi-Maaruf. Dukungan Mbah Moen diyakininya juga akan berdampak pada dukungan dari murid-murid kiai karismatik itu.  #2019JokowiKyaiMaruf #2019TetapJokowi #TetapJokowi #IndonesiaMaju #Jokowi #jokowidodo #indonesiajuara #MarufAmin #JokowiAmin #Jokowi1KaliLagi #JokowiLagi #pilpres #pilpres2019 #GenerasiOptimis #4TahunJokowi']

final = []
i=0
for caption in captions  :
	x=clean_instagram(caption).lower()
	clear = stopword(x)
	clear2 = stemming(clear)
	clear2 = word_tokenize(clear2)
	
	# print (clear2)
	final.append(clear2)
	i+=1
	print('\r{}/{} completed\r'.format(i, len(captions)), end="")

print('')
labels = data['Class'].values

print(np.array(final).shape)
print(np.array(labels).shape)

res = np.vstack([labels, final])
# res2 = np.hstack([labels, final])
# print(res.shape)
# print(res[0])
# print(res[1])

# print(res2.shape)

# res = np.hstack([np.array(final).T, np.array(labels).T])
# print(res)
# print(res.shape)
# res = [{'Text':final}, labels]
# print(res)

frame = pd.DataFrame(res.T, columns=['Class', 'Text'])
frame.to_csv("hasilpreproc.csv", index=False)


# idea={'caption':final}
# frame=pd.DataFrame(idea, columns=['caption']) 
# print(frame)
# frame.to_csv("hasiltokenize1.csv", index=False)
