import numpy as np
import pickle
# import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize

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

def load_train_data(filename):
	res = 0
	try:
		out = open(filename, 'rb')
		res = pickle.load(out)
		out.close()
	except:
		print('Something went wrong or file is not available')

	if res == 0:
		return False

	return res

def find_train_result(filename, word):
	result = load_train_data(filename)

	if result:
		if word in result.keys():
			return result[word]
		else:
			return False
	else:
		return False

def test(sentence):
	x=clean_instagram(sentence).lower()
	clear = stopword(x)
	clear2 = stemming(clear)
	clear2 = word_tokenize(clear2)

	model = load_train_data('training_result.dat')

	result = [1,1,1]

	for x in range(0,3):
		for word in clear2:
			# print(word)
			for (key, value) in model[word].items():
				# print(key)
				if key == 'pab'+str(x-1):
					# print(x-1)
					# print(value)
					result[x] *= value

	return result.index(max(result))-1, max(result)

def load_test_file(filename, sep=','):
	data = pd.read_csv(filename, sep=sep)
	dataLen = len(data)
	valTrue = 0
	for index, row in data.iterrows():
		res, val = test(row['Text'])
		if res == row['Class']:
			print("{} is same with {:2f}".format(index, val))
			valTrue+=1
		else:
			print("{} is not same with value {:2f} and answer {}".format(index, val, res))

	print('accuracy', valTrue*1.0/dataLen)
		# print('result', test(row['Text']))
		# print('real idx', row['Class'])

# print(test('repost from jokowi'))
load_test_file('DATA TRAINING.csv', ';')