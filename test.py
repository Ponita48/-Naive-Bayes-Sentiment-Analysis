import numpy as np
import pickle
# import matplotlib.pyplot as plt
import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

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

def do_preproc(sentence):
	x=clean_instagram(sentence).lower()
	clear = stopword(x)
	clear2 = stemming(clear)
	clear2 = word_tokenize(clear2)

	return clear2

def test(sentence):
	clear2 = do_preproc(sentence)
	model = load_train_data('training_result.dat')

	result = [1,1,1,1,1,1]

	for x in range(-3,3):
		for word in clear2:
			# print(word)
			if word in model.keys():
				for (key, value) in model[word].items():
					# print(key)
					if key == 'pab'+str(x):
						# print(x-1)
						# print(value)
						result[x] *= value

	return result.index(max(result))-1, max(result)

def count_accuracy(filename):
	data = pd.read_csv(filename, sep=";")
	val = data['Class'].values
	text = data['Text'].values

	result = []
	matching = 0

	for sentence in text:
		res, _ = test(sentence)
		result.append(res)

	for x in range(len(val)):
		if val[x] == result[x]:
			matching+=1

	return matching/len(val)

print('{:2f}'.format(count_accuracy('DATA TRAINING.csv')))

# print(test('Tukang Bakso juga milih Prabowo-Sandi ! #prabowo #sandiagauno #prabowosandi #2019gantipresiden #pks #pan #gerindra #partaiberkarya #partaidemokrat #2019prabowopresiden #2019presidenbaru #gantipresiden #gantipresiden2019'))