import numpy as np
import pandas as pa
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
#import gensim
import os
from gensim import models,corpora,similarities
from gensim.models import LdaModel
from nltk import FreqDist
from scipy.stats import entropy
from nltk.tokenize import TweetTokenizer,word_tokenize
warnings.filterwarnings('ignore')
sn.set_style("darkgrid")

### Read all the data cleaned
filedir = os.path.abspath(r"C:\Users\BABI\Dynamic Blog Recommendation\Cleaned Data")
medium_filename = "cleaned_medium"
ana_filename = "cleaned_analytics"
toward_filename = "cleaned_towards_data_science"

toward_filepath = os.path.join(filedir,toward_filename)
medium_filepath = os.path.join(filedir,medium_filename)
ana_filepath = os.path.join(filedir,ana_filename)

data_medium = pa.read_csv(medium_filepath)
data_medium['Webpage'] = 'Medium'
data_toward = pa.read_csv(toward_filepath)
data_toward['Webpage'] = 'Towards_Data_Science'
data_toward = data_toward.rename(columns={'Link':'Links'})
data_ana = pa.read_csv(ana_filepath)
data_ana = data_ana.rename(columns={'Titles':'Title'})
data_ana['Webpage'] = 'Analytics_Vidhya'
data = pa.concat([data_medium,data_toward])
data = pa.concat([data,data_ana])
data.reset_index(drop=True,inplace=True)

tokenizer = TweetTokenizer()
data_words = data['Description'].apply(lambda x:tokenizer.tokenize(x))
all_words = [word for item in data_words for word in item]
data['tokenized'] = data_words

# Frequency dist of all words
fdist = FreqDist(all_words)

k=20000
top_words = fdist.most_common(k)
print('Last Top Words',top_words[-10:])
print('First Top Words',top_words[0:10])

top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
def store_topwords(words):
    words= [word for word in words if word in top_k_words]
    return words
data['tokenized'] = data['tokenized'].apply(lambda x:store_topwords(x))

data = data[data['tokenized'].map(len) > 30]
data = data[data['tokenized'].map(type)==list]
data.reset_index(drop=True,inplace=True)
data = data.drop(columns=['Unnamed: 0','Unnamed: 0.1'],axis=1)
print('Data Shape',data.shape)

## Training the lda model

#mask = np.random.rand(len(data)) < 0.999
#train_data  = data[mask]
train_data = data

#train_data.to_csv('training_data.csv')

test_data = train_data.iloc[230]#give a blog to predict
train_data = train_data.drop(index=230,axis=0)
#train_data.reset_index(drop=True,inplace=True)
#test_data = data[~mask]
#test_data.reset_index(drop=True,inplace=True)

train_data.reset_index(drop=True,inplace=True)

#Model (latent Dirichlet Allocation)
def lda_model(train_data):
    num_topics = 8
    chunksize = 200
    dictionary = corpora.Dictionary(train_data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in train_data['tokenized']]
    
    #Define the lda model
    lda = LdaModel(corpus=corpus,num_topics=num_topics,id2word=dictionary,alpha=0.8e-1,eta=0.03e-3,chunksize=chunksize,
                  minimum_probability=0.0,passes=2)
    return dictionary,corpus,lda

dictionary,corpus,lda = lda_model(data)

### Topic Visulalization
lda.show_topics(num_topics=20,num_words=20)

train_rand = np.random.randint(len(train_data))
bow = dictionary.doc2bow(train_data['tokenized'].iloc[train_rand])
## Topic distribution of that particular document 
doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
patches = ax.bar(np.arange(len(doc_distribution)), doc_distribution)
ax.set_xlabel('Topic ID', fontsize=15)
ax.set_ylabel('Topic Contribution', fontsize=15)
ax.set_title("Topic Distribution for Article " + str(train_rand), fontsize=20)
fig.tight_layout()
plt.show()

##Visualize the cluster of this particular document
for i in doc_distribution.argsort()[::-1][:7]:
    print(i,lda.show_topic(topicid=i,topn=10),"\n")
    
    
### Simlilarity for unseen data
### Select random from test data
test_data_index = np.random.randint(len(test_data))
test_bow = dictionary.doc2bow(test_data['tokenized'])

test_data['Links']
### Check the topic distribution
test_topic_dist = np.array([tup[1]  for tup in lda.get_document_topics(bow=test_bow)]) 
## Topic distribution of that particular document 
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
patches = ax.bar(np.arange(len(test_topic_dist)), test_topic_dist)
ax.set_xlabel('Topic ID', fontsize=15)
ax.set_ylabel('Topic Contribution', fontsize=15)
ax.set_title("Topic Distribution for Article " + str(test_data_index), fontsize=20)
fig.tight_layout()
plt.show()
##Visualize the cluster of this particular document


# Find similer topics usning jensen_shannon distance 
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
def jensen_shannon(query, matrix):
    p = query[None,:].T 
    q = matrix.T
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=10): #k=10 it will recommend top 10 documents simliar to the given document
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances

## Find the docuemt whiich are similar to the test document
most_sim_ids = get_most_similar_documents(test_topic_dist,doc_topic_dist)

most_sim_documents = train_data[train_data.index.isin(most_sim_ids)]

for i in range(0,10):
    print('Title {}"\n"Link--{}'.format(most_sim_documents['Title'].iloc[i],
                                      most_sim_documents['Links'].iloc[i]))
    print('******************************')
    

#import pickle
#pickle.dump(lda,open('lda_model','wb')) ## Store the model
#pickle.dump(dictionary,open('dictonary','wb')) ## Store the dictonary
#pickle.dump(corpus,open('corpus','wb')) ## Store tje corpus 



