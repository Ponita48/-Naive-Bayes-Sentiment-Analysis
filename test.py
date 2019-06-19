import numpy as np
import pickle
# import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
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

def do_preproc(sentence):
	x=clean_instagram(sentence).lower()
	clear = stopword(x)
	clear2 = stemming(clear)
	clear2 = word_tokenize(clear2)

	return clear2



def find_train_result(filename, word):
	result = load_train_data(filename)

	if result:
		if word in result.keys():
			return result[word]
		else:
			return False
	else:
		return False

def test(sentence, traindata):
	clear2 = do_preproc(sentence)
	model = load_train_data(traindata)

	result = [1,1,1]

	for x in range(-1,2):
		for word in clear2:
			# print(word)
			if word in model.keys():
				for (key, value) in model[word].items():
					# print(key)
					if key == 'pab'+str(x):
						# print(x-1)
						# print(value)
						result[x] += value

	return result.index(max(result))-1, max(result)
	

def count_accuracy(filename, trainname):
	data = pd.read_csv(filename, sep=";")
	val = data['Class'].values
	text = data['Text'].values

	result = []
	matching = 0

	for sentence in text:
		res, _ = test(sentence, trainname)
		result.append(res)

	for x in range(len(val)):
		if val[x] == result[x]:
			matching+=1

	return matching/len(val)

# print('data1: {:2f}'.format(count_accuracy('DATA TRAINING.csv', 'training_result.dat')))
# print('data2: {:2f}'.format(count_accuracy('Data training 02.csv', 'training_result2.dat')))


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
print('{:2f}'.format(count_accuracy('DATA TRAINING.csv')))
load_test_file('DATA TRAINING.csv', ';')
