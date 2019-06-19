#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Doc2Vec
import re, string, collections, pickle, os
import pandas as pd
import numpy as np
import multiprocessing
import ipython_genutils
import progressbar
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import WordPunctTokenizer
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import utils
from sklearn.utils import shuffle
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import Tokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.utils import to_categorical
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
import utils
import importlib
importlib.reload(utils)
from utils import *


# In[2]:


df = pd.read_csv('lib/dataset',index_col=0)
df1 = pd.read_csv('lib/dataset ahok',index_col=0)
df2 = pd.read_csv('lib/datasetfm',index_col=0)
df=pd.concat([df,df1,df2],sort=False)
#df['target']=df['target'].map({0:-1,1:0,2:1})
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df = shuffle(df)
df.info()
df.head()


# In[3]:


df.groupby('target')['text'].count()


# In[4]:


dneg=df.query('target == -1')
dnet=df.query('target == 0')
dpos=df.query('target == 1')

minlen=min(len(dnet.index),len(dpos.index),len(dneg.index))


dnet=dnet.head(minlen)
dneg=dneg.head(minlen)
dpos=dpos.head(minlen)

data_training=pd.concat([dneg.text,dnet.text,dpos.text])
target=pd.concat([dneg.target,dnet.target,dpos.target])
target=to_categorical(target-target.min())


# In[5]:


df_ds=pd.DataFrame({'text':data_training,'target':np.argmax(target, axis=1, out=None)})
df_ds.groupby('target')['text'].count()


# In[6]:



X_train, X_test, y_train, y_test = model_selection.train_test_split(df_ds['text'].values, df_ds['target'].values, test_size=0.2, random_state=42)
print('data train\t: ',len(X_train))
print('data validasi\t: ',len(X_test),)


# In[7]:


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[8]:


vect = CountVectorizer(tokenizer=tokenize)


# In[9]:


tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)


# In[10]:


tf_train


# In[11]:


vocab = vect.get_feature_names()
len(vocab)


# In[12]:


vocab[4000: 4005]


# In[13]:


w0 = set([o for o in X_train[0].split(' ')])


# In[14]:


w0


# In[15]:


vect.vocabulary_['unless']


# In[16]:


tf_train[0, 20341]


# In[17]:


from sklearn.decomposition import TruncatedSVD, PCA


# In[18]:


svd = TruncatedSVD()
reduced_tf_train = svd.fit_transform(tf_train)


# In[19]:


plot_embeddings(reduced_tf_train, y_train)


# In[26]:


p = tf_train[y_train==1].sum(0) + 1
q = tf_train[y_train==0].sum(0) + 1
r = tf_train[y_train==-1].sum(0) + 1
s = np.log((p/p.sum())/(q/q.sum())/(r/r.sum()))
b = np.log(len(p)/len(q)/len(r))


# In[27]:


pre_preds = tf_test @ s.T + b
preds = pre_preds.T > 0
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')


# In[ ]:
