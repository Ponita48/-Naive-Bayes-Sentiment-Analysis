import numpy as np
import pandas as pd
import ast
import math

class Evidence:

	def __init__(self, data, kelas):
		self.data = data
		self.wordsCount = {}
		self.words = []
		self.className = kelas
		self.length = self.findLengthClass()

	def tokenize(self, sentence):
		y = ast.literal_eval(sentence)
		y = [n.strip() for n in y]

		return y

	def findLengthClass(self):
		arr = np.array(self.data['Class'].values)
		condition = np.isin(arr, self.className)
		value = np.where(condition)
		result = 0
		for x in value[0]:
			self.addData(self.data.iloc[x]['Text'])
			y = self.tokenize(self.data.iloc[x]['Text'])

			result += len(y)

		return result

	def addData(self, sentence):
		words = self.tokenize(sentence)
		for x in words:
			if x in self.wordsCount:
				self.wordsCount[x] += 1
			else:
				self.wordsCount[x] = 1
				self.words.append(x)

	def getProbability(self, word):
		count = self.wordsCount[word]
		return count/self.length

	def getData(self):
		return self.data

	def getWordsCount(self):
		return self.wordsCount

	def getWords(self):
		return self.words

	def getClassName(self):
		return self.className

	def getLength(self):
		return self.length
	
class Classifier:

	def __init__(self, filename):
		self.filename = filename
		self.data = pd.read_csv(filename, sep=',', header=0)
		self.posResp = Evidence(self.data, 1)
		self.netResp = Evidence(self.data, 0)
		self.negResp = Evidence(self.data, -1)
		self.bow = []
		self.score = {'1':0, '0':0, '-1':0}
		self.mergeAllWords()
		self.countTotalWords()
		self.countScore()

	def mergeAllWords(self):
		self.bow.extend(self.posResp.getWords())
		self.bow.extend(self.netResp.getWords())
		self.bow.extend(self.negResp.getWords())
		self.bow = np.unique(self.bow).tolist()


	def countTotalWords(self):
		self.length = self.posResp.getLength() + self.netResp.getLength() + self.negResp.getLength()

	def countScore(self):
		for x in self.bow:
			self.score['1'] += float(float(math.log(self.posResp.getWordsCount().get(x, 1)) / self.posResp.getLength()))
			self.score['0'] += float(float(math.log(self.netResp.getWordsCount().get(x, 1)) / self.netResp.getLength()))
			self.score['-1'] += float(float(math.log(self.negResp.getWordsCount().get(x, 1)) / self.negResp.getLength()))
			# print(self.score['1'], self.score['0'], self.score['-1'])
			# print(self.posResp.getWordsCount().get(x,1), self.posResp.getLength(), math.log(self.posResp.getWordsCount().get(x,1)/self.posResp.getLength()))
		
		self.score['1'] += float(math.log(self.posResp.getLength() / self.length))
		self.score['0'] += float(math.log(self.netResp.getLength() / self.length))
		self.score['-1'] += float(math.log(self.negResp.getLength() / self.length))

		self.score['1'] = float(math.exp(self.score['1']))
		self.score['0'] = float(math.exp(self.score['0']))
		self.score['-1'] = float(math.exp(self.score['-1']))

		totalScore = self.score['1'] + self.score['0'] + self.score['-1']

		self.score['1'] = float((100*self.score['1'])/totalScore)
		self.score['0'] = float((100*self.score['0'])/totalScore)
		self.score['-1'] = float((100*self.score['-1'])/totalScore)

	def classify(self, sentence, tokenize=True):
		if tokenize:
			words = self.tokenize(sentence)
		else:
			words = sentence
		score = {'1':0, '0':0, '-1':0}

		for x in words:
			score['1'] += float(float(math.log(self.posResp.getWordsCount().get(x, 1)) / self.posResp.getLength()))
			score['0'] += float(float(math.log(self.netResp.getWordsCount().get(x, 1)) / self.netResp.getLength()))
			score['-1'] += float(float(math.log(self.negResp.getWordsCount().get(x, 1)) / self.negResp.getLength()))
		
		score['1'] += float(float(math.log(self.posResp.getLength()) / self.length))
		score['0'] += float(float(math.log(self.netResp.getLength()) / self.length))
		score['-1'] += float(float(math.log(self.negResp.getLength()) / self.length))

		score['1'] = float(math.exp(score['1']))
		score['0'] = float(math.exp(score['0']))
		score['-1'] = float(math.exp(score['-1']))

		totalScore = score['1'] + score['0'] + score['-1']

		score['1'] = float((100*score['1'])/totalScore)
		score['0'] = float((100*score['0'])/totalScore)
		score['-1'] = float((100*score['-1'])/totalScore)

		return score

		
	def tokenize(self, sentence):
		y = ast.literal_eval(sentence)
		y = [n.strip() for n in y]

		return y

	def getFilename(self):
		return self.filename

	def getBow(self):
		return self.bow
	
	def getLength(self):
		return self.length

	def getScore(self):
		return self.score

		

# data = pd.read_csv('hasilpreproc.csv', sep=',', header=0)
# jokowiNet = Evidence(data, 0)
# jokowiPos = Evidence(data, 1)
# jokowiNeg = Evidence(data, -1)

# data2 = pd.read_csv('hasilpreproc2.csv', sep=',', header=0)
# praNet = Evidence(data2, 0)
# praPos = Evidence(data2, 1)
# praNeg = Evidence(data2, -1)

# print(jokowiNet.getWordsCount())

fn1 = 'hasilpreproc.csv'
fn2 = 'hasilpreproc2.csv'

clas1 = Classifier(fn1)
clas2 = Classifier(fn2)

# print('cls1', clas1.getScore())
# print('cls2', clas2.getScore())

# import numpy as np
import pickle
# import matplotlib.pyplot as plt
import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
import operator
# import pandas as pd

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

data = pd.read_csv('DATA TRAINING.csv', sep=";")
val = data['Class'].values
text = data['Text'].values
index = 0
count = 0

for sentence in text:
	print('({}/{}) words, {:.2f}% completed\r'.format(index, len(text), 100*(index/(len(text)))), end="")
	x = do_preproc(sentence)
	# print(x[0])
	result = clas1.classify(x, False)
	resultIndex = max(result.items(), key=operator.itemgetter(1))[0]
	# print()
	# print(resultIndex, val[index])
	# print(int(resultIndex) == int(val[index]))
	if int(resultIndex) == int(val[index]):
		# print('plusplus')
		count += 1

	index += 1

print("")
print(100*count/index)