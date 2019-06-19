import numpy as np
import pandas as pd
import ast

# hitung jumlah kata yang ada di dataset
def findLength(data):
	total = 0
	for x in data:
		y = ast.literal_eval(x)
		y = [n.strip() for n in y]

		total += len(y)

	return total

def findLengthClass(data, find):
	arr = np.array(data['Class'].values)
	condition = np.isin(arr, find)
	value = np.where(condition)
	result = 0
	for x in value[0]:
		y = ast.literal_eval(data.iloc[x]['Text'])
		y = [n.strip() for n in y]

		result += len(y)

	return result

# cari dimana sebuah kata keluar pada dataset
def findOccurences(dataset, word):
	occur = []
	for x in range(len(dataset)):
		data = ast.literal_eval(dataset[x])
		data = [n.strip() for n in data]
		for y in range(len(data)):
			if data[y] == word:
				occur.append([x, y])

	return occur

# p(A) <words>
# hitung probabilitas sebuah kata pada seluruh dataset (jumlah kata/jumlah dataset)
def findProbX(data, word):
	occur = findOccurences(data['Text'].values, word)
	length = findLength(data['Text'].values)
	return (len(occur)*1.0)/length, len(occur)

# p(B) <class>
# hitung probabilitas sebuah class pada seluruh dataset (jumlah kelas/jumlah dataset)
def findProbClass(data, find):
	arr = np.array(data['Class'].values)
	unique, counts = np.unique(arr, return_counts=True)
	final = dict(zip(unique, counts))
	occur = final[find]
	length = len(data['Text'].values)
	return (occur*1.0)/length*1.0, occur

# p(B|A)
# hitung probabilitas kata keluar pada class tertentu (jml kata/jml kelas)
def findClass(data, word, find):
	words = np.array(data['Text'].values)
	occur = findOccurences(words, word)
	counter = 0
	for x in occur:
		clas = data.iloc[x[0]]['Class'] 
		if clas == find:
			counter += 1
	_, kelas = findProbClass(data, find)
	# print(counter, kelas)
	return (counter*1.0)/kelas*1.0, counter

# mencari hasil akhir
def findProb(data, word, find):
	pba,_ = findClass(data, word, find)
	pa,_ = findProbX(data, word)
	pb,_ = findProbClass(data, find)

	return pba*pa/pb

# gabungkan 2 dataset
def mergeTrainData(data1, data2):
	data2 = data2.replace({'Class':-1}, -2)
	data2 = data2.replace({'Class':0}, -3)
	data2 = data2.replace({'Class':1}, 2)

	data = [data1, data2]

	data = pd.concat(data)

	return data

data = pd.read_csv('hasilpreproc.csv', sep=',', header=0)

words = data['Text'].values

bow = []
for x in words:
	x = ast.literal_eval(x)
	x = [n.strip() for n in x]
	for y in x:
		# print(y not in bow)
		if y not in bow:
			bow.append(y)

final_prob = {}

i = 1
for word in bow:
	print('learning ({}/{}) words, {:.2f}% completed\r'.format(i, len(bow), 100*(i/(len(bow)))), end="")
	this_prob = {}
	for x in range(-1, 2):
		result = findProb(data, word, x)
		this_prob['pab'+str(x)] = result
	final_prob[word] = this_prob
	i+=1

# Save ke pickle jadi gausah learning lagi nanti
import pickle

pickle_out = open("training_result.dat", "wb")
pickle.dump(final_prob, pickle_out)
pickle_out.close()

data2 = pd.read_csv('hasilpreproc2.csv', sep=',', header=0)

words = data['Text'].values

bow = []
for x in words:
	x = ast.literal_eval(x)
	x = [n.strip() for n in x]
	for y in x:
		# print(y not in bow)
		if y not in bow:
			bow.append(y)

final_prob = {}

i = 1
for word in bow:
	print('learning ({}/{}) words, {:.2f}% completed\r'.format(i, len(bow), 100*(i/(len(bow)))), end="")
	this_prob = {}
	for x in range(-1, 2):
		result = findProb(data, word, x)
		this_prob['pab'+str(x)] = result
	final_prob[word] = this_prob
	i+=1

# Save ke pickle jadi gausah learning lagi nanti

pickle_out = open("training_result2.dat", "wb")
pickle.dump(final_prob, pickle_out)
pickle_out.close()


# data2 = pd.read_csv('hasilpreproc2.csv', sep=',', header=0)

# data = mergeTrainData(data, data2)
# for x in range(-3, 3):
# 	print('len cls',x,':',findLengthClass(data, x))

# print(data.keys())

# print(final_prob)

# occur = findOccurences(words, 'repost')
# length = findLength(words)

# print(len(occur), '/', length)
# print(findProbX(occur, length))

# print('pba', findClass(data, 'repost', 1))
# print('pa', findProbX(data, 'repost'))
# print('pb', findProbClass(data, 1))