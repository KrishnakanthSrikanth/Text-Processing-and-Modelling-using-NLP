#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Krishnakanth Srikanth
# #### Student ID: s3959200
# 
# Date: 01/10/2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# * CountVectorizer
# * TfidfVectorizer
# * genism
# * api
# * train_test_split
# * LogisticRegression
# * FreqDist
# * KFold
# * RegexpTokenizer
# * sent_tokenize
# * chain
# 
# ## Introduction
# In this task, feature representation of documents such as count vector, weighted and unweighted TF-IDF vector are to be calculated for job description. Using which Logistic Regression model is built, and the best vector representation is found out.
# Additionally, a question if adding more information like title to the description, affects the model accuracy or not is answered in here.

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#pip install gensim
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.probability import FreqDist
from sklearn.model_selection import KFold
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# 3 different types of feature representation of documents are to be built in this task - count vector, two document embeddings (one TF-IDF weighted, and one unweighted version).

# ###### Generating Count vector using Bag-of-words model

# In[2]:


# Loading the vocab.txt file from task 1 to train BOW model
# Empty list to append words
vocab_list = []

with open("vocab.txt") as f:
    vocab = f.readline()
    while vocab:
        end = vocab.find(":")
        vocab_list.append(vocab[:end])
        vocab = f.readline()     


# In[3]:


# Regenerating the job descriptions from task 1
with open("job_desc.txt", "r") as f:
    job_desc = f.read().splitlines()
job_desc = [desc.split(" ") for desc in job_desc]


# ###### Count vector generation

# In[4]:


# Count vector generation
joined_job_desc = [' '.join(desc) for desc in job_desc]
cVectorizer = CountVectorizer(analyzer = "word", vocabulary = vocab_list)
count_features = cVectorizer.fit_transform(joined_job_desc)
count_features.shape[0]


# In[5]:


# Check for count_features to see if the output is as expected
print('JD 1:' + '\n' + joined_job_desc[0] + '\n\n' + 'Count vector for JD 1: ')
for word, value in zip(vocab_list, count_features.toarray()[0]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# In[6]:


# Loading the web_index from task1
web_indices = []
with open("web_indices.txt", "r") as f:
    # -1 to remove later spaces
    idx = f.readline()[:-1]
    while idx:
        web_indices.append(idx)
        idx = f.readline()[:-1]


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[7]:


def write_count_vector(count_vector, filename, web_indices):
    '''
        This function is to store the count vectors to a new text file
    '''
    # total number of documents
    num = count_vector.shape[0] 
    
    # create a new txt file if it doesn't exists 
    cvector_file = open(filename, 'w') 
    
    for i in range(0, num): 
        web_index = web_indices[i]
        cvector_file.write("#" + web_index + ",")
        
        # for each word index that has non-zero entry in the count_vector
        for j in count_vector[i].nonzero()[1]: 
            
            # retrieve the value of the entry from count_vector
            value = count_vector[i][0, j]
            
            if int(np.where(count_vector[i].nonzero()[1] == j)[0]) == len(count_vector[i].nonzero()[1])-1:
                cvector_file.write("{}:{} ".format(j, value))
                
            else:
                # write the entry to the file in the format of word_index:value
                cvector_file.write("{}:{},".format(j, value)) 
        
        cvector_file.write('\n') 
    
    # close the file
    cvector_file.close()
    
write_count_vector(count_features,"count_vectors.txt", web_indices)


# ###### Models based on word embeddings

# In[8]:


# generate TF-IDF vectors :
from sklearn.feature_extraction.text import TfidfVectorizer

# initialised the the TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word", vocabulary = vocab_list)

# generate the TF-IDF vector representation for all description
tfidf_features = tVectorizer.fit_transform(joined_job_desc) 

tfidf_features.shape


# In[9]:


def gen_docVecs_weighted(wv,tk_txts,tfidf = []):
    '''
        Function to generate weighted vector representations for documents
    '''
    docs_vectors = pd.DataFrame()

    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # Distinct words of document is collected using set()

        temp = pd.DataFrame()
        for w_ind in range(0, len(tokens)):
            try:
                word = tokens[w_ind]
                word_vec = wv[word]
                
                if tfidf != []:
                    word_weight = float(tfidf[i][word])
                else:
                    word_weight = 1
                temp = temp.append(pd.Series(word_vec*word_weight), ignore_index = True)
            except:
                pass
        doc_vector = temp.sum()
        # append each document value to the final dataframe
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True)
    return docs_vectors


# In[10]:


# method to generate vector representation for documents - unweighted
def gen_docVecs_unweighted(wv, tk_txts): 
    '''
        Function to generate unweighted vector representations for documents
    '''
    # creating empty final dataframe
    docs_vectors = pd.DataFrame()

    for i in range(0, len(tk_txts)):
        tokens = tk_txts[i]
        
        temp = pd.DataFrame() 
        
        for w_ind in range(0, len(tokens)): 
            try:
                word = tokens[w_ind]
                
                 # if word is present in embeddings then proceed
                word_vec = wv[word]
                
                temp = temp.append(pd.Series(word_vec), ignore_index = True) 
            except:
                pass
        
        # sum across rows of each column
        doc_vector = temp.sum() 
        
        # append each document value to the final dataframe
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) 
        
    return docs_vectors


# In[11]:


def gen_vocIndex(voc_fname):
    '''
        This function reads the the vocabulary file, and create an w_index:word dictionary
    '''
    with open(voc_fname) as vocf: 
        voc_Ind = [l.split(':') for l in vocf.read().splitlines()] # each line is 'index,word'
    return {int(vi[1]):vi[0] for vi in voc_Ind}

# Generates the w_index:word dictionary
voc_fname = 'vocab.txt' # path for the vocabulary
voc_dict = gen_vocIndex(voc_fname)


# In[12]:


# checking for the weights 
num = tfidf_features.shape[0] # the number of document
tfidf_weights =[]

for wt in range(0, num): 
    weight_dict = {}
    for word, value in zip(vocab_list, tfidf_features.toarray()[wt]): 
        if value > 0:
            weight_dict[word] = value
    tfidf_weights.append(weight_dict)


# ###### Embedding language model - GoogleNews300

# In[13]:


# Load word2vec google news 300 api
google_api = api.load('word2vec-google-news-300')


# In[14]:


# Writing the job descriptions to txt file
with open("jd_task1.txt", "w") as file:
    for i in range(0, len(joined_job_desc)):
        file.write(joined_job_desc[i] + "\n")


# In[15]:


# tokenization of description
with open('jd_task1.txt') as file:
    desc_text = file.read().splitlines() 
tokenized_description = [a.split(' ') for a in desc_text]


# ###### TF-IDF Unweighted representation

# In[16]:


# TF-IDF Unweighted representation
unweighted_idf = gen_docVecs_unweighted(google_api, pd.Series(tokenized_description))
unweighted_idf.shape[0]


# ###### TF-IDF Weighted representation

# In[17]:


# TF-IDF Weighted representation
weighted_idf = gen_docVecs_weighted(google_api, pd.Series(tokenized_description),tfidf_weights)
weighted_idf.shape[0]


# ## Task 3. Job Advertisement Classification

# ###### Q1: Language model comparisons

# In[18]:


# KFold cross validation with 5 folds
from sklearn.model_selection import KFold
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=0, shuffle = True)
print(kf)


# In[19]:


def evaluate(X_train,X_test,y_train, y_test,seed):
    '''
        Function to build Logistic Regression model on features created
    '''
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[20]:


target = []
with open('./job_category_target.txt') as f: 
    target = f.readlines()
target=[i.strip('\n') for i in target]


# In[21]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
seed=0
num_models = 2
model_df = pd.DataFrame(columns = ['count', 'weighted_idf','unweighted_idf'], index=range(num_folds))
fold = 0
for train_index, test_index in kf.split(list(range(0,len(target)))):
    y_train = [str(target[i]) for i in train_index]
    y_test = [str(target[i]) for i in test_index]
    
    X_train_count, X_test_count = count_features[train_index], count_features[test_index]
    model_df.loc[fold,'count'] = evaluate(count_features[train_index],count_features[test_index], y_train, y_test, seed)
    
    X_train_wt, X_test_wt = weighted_idf.iloc[train_index, :-1], weighted_idf.iloc[test_index, :-1]
    model_df.loc[fold,'weighted_idf'] = evaluate(X_train_wt, X_test_wt, y_train, y_test, seed)
    
    X_train_uwt, X_test_uwt = unweighted_idf.iloc[train_index, :-1], unweighted_idf.iloc[test_index, :-1]
    model_df.loc[fold,'unweighted_idf'] = evaluate(X_train_uwt, X_test_uwt, y_train, y_test, seed)
    
    fold +=1
    
model_df


# In[22]:


# Model Evaluation
model_df.mean()


# From the above results, it is clear that __COUNT VECTOR FEATURE REPRESENTATION__ performs the best followed by __WEIGHTED TF-IDF FEATURE REPRESENTATION__ with the Logistic Regression model.

# ###### Q2: Does more information provide higher accuracy?

# ###### With only Title of the job advertisement

# In[23]:


# Read the titles file created in task 1
titles = []
with open("job_titles.txt", "r") as file:
    title = file.readline()
    while title:
        titles.append(title[:-1])
        title = file.readline() 


# In[24]:


# Tokenize the titles
def tokenizeData(raw_data):
    # cover all words to lowercase
    nl_data = raw_data.lower()
    
    # segment into sentences
    sentences = sent_tokenize(nl_data)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_data = list(chain.from_iterable(token_lists))
    return tokenised_data

tokenized_job_titles = [tokenizeData(title) for title in titles]


# In[25]:


# Remove words with length less than 2
tk_job_titles_g2 = [[token for token in title if len(token) >= 2] for title in tokenized_job_titles]


# In[26]:


stopwords_list = []
with open('./stopwords_en.txt') as f:
    stopwords_list = f.read().splitlines()

# remove stop words
tk_job_titles_stp = [[word for word in job if word not in stopwords_list] for job in tk_job_titles_g2]


# In[27]:


# Remove words that appear only once by term frequency
words = list(chain.from_iterable(tk_job_titles_stp))

# compute term frequency for each unique word/type
term_freq = FreqDist(words)
lessFreqWords = set(term_freq.hapaxes())

def removeLessFreqWords(words):
    '''
        This function is to remove the words that appear only once in document based on term frequency
    '''
    return [w for w in words if w not in lessFreqWords]

tk_removeLessTermFreq = [removeLessFreqWords(word) for word in tk_job_titles_stp]


# In[28]:


# Remove the top 50 most frequent words by document frequency
words_2 = list(chain.from_iterable([set(tk) for tk in tk_removeLessTermFreq]))

# find words that appear most commonly across documents
doc_freq = FreqDist(words_2)  
doc_freq_sorted = sorted(list(doc_freq.most_common(50)))

# Creating a list to append the top 50 words
doc_freq_words = []
for i,j in doc_freq_sorted:
    doc_freq_words.append(i)

def removeTop50(words):
    '''
        This function is to remove top 50 frequent words based on document frequency
    '''
    return [word for word in words if word not in doc_freq_words]

tk_removeMostDocumentFreq = [removeTop50(words) for words in tk_removeLessTermFreq]


# In[29]:


tokenized_titles = [" ".join(token) for token in tk_removeLessTermFreq]


# In[30]:


# Generating Count vector representation for title
jobTitle = list(chain.from_iterable(tk_removeLessTermFreq))
title_vocab = set(jobTitle)
cVectorizerTitle = CountVectorizer(analyzer = "word", vocabulary = title_vocab)

# fit the model on job descriptions
count_features_title = cVectorizerTitle.fit_transform(tokenized_titles)


# In[31]:


# Generating TFIDF vector for generating weighted and unweighted word embeddings of decriptions
tVectorizerTitle = TfidfVectorizer(analyzer = "word", vocabulary = title_vocab)
tfidf_features_title = tVectorizerTitle.fit_transform(tokenized_titles)


# In[32]:


# Saving the tokenized job titles for later use
with open("tk_titles.txt", "w") as file:
    for i in range(0, len(tokenized_titles)):
        file.write(tokenized_titles[i] + "\n")


# In[33]:


filename = 'tk_titles.txt'
with open(filename) as f:
    title = f.read().splitlines()
tk_titles_list = [a.split(' ') for a in title]


# In[34]:


# TF-IDF Unweighted representation
unweighted_idf_title = gen_docVecs_unweighted(google_api, pd.Series(tk_titles_list))
unweighted_idf_title.fillna(0.0, inplace = True) # Replacing missing values with 0
unweighted_idf_title.shape[0]


# In[35]:


# TF-IDF Weighted representation
weighted_idf_title = gen_docVecs_weighted(google_api, pd.Series(tk_titles_list),tfidf_weights)
weighted_idf_title.fillna(0.0, inplace = True) # Replacing missing values with 0
weighted_idf_title.shape[0]


# In[36]:


# Model with only title - Unweighted
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_idf_title, target, list(range(0,len(target))),test_size=0.33, random_state=0)
model = LogisticRegression(max_iter = 100, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[37]:


# Model with only title - Weighted
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(weighted_idf_title, target, list(range(0,len(target))),test_size=0.33, random_state=0)
model = LogisticRegression(max_iter = 100, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ###### With only Description of the job advertisement

# In[38]:


# Model with only description - Unweighted
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_idf, target, list(range(0,len(target))),test_size=0.33, random_state=0)
model = LogisticRegression(max_iter = 100, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[39]:


# Model with only description - Weighted
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(weighted_idf, target, list(range(0,len(target))),test_size=0.33, random_state=0)
model = LogisticRegression(max_iter = 100, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ###### With both the Title and Description of the job advertisement

# In[40]:


# Concatenate title and description of each job advertisement and add to a list
title_desc = []
for i in range(0,len(target)):
    tit_des = " ".join(tk_titles_list[i]) + " " + joined_job_desc[i]
    title_desc.append(tit_des)


# In[41]:


# Tokenize title_desc
tk_title_desc = [tokenizeData(job) for job in title_desc]  


# In[42]:


# Count vector generation
words = list(chain.from_iterable(tk_title_desc))
tit_desc_vocab = sorted(list(set(words)))
cVectorizer = CountVectorizer(analyzer = "word", vocabulary = tit_desc_vocab)
count_features_titdesc = cVectorizer.fit_transform(title_desc)
count_features_titdesc.shape[0]


# In[43]:


# TF-IDF generation
tVectorizer = TfidfVectorizer(analyzer = "word", vocabulary = tit_desc_vocab) 
tfidf_features_titdesc = tVectorizer.fit_transform(title_desc) 
tfidf_features_titdesc.shape[0]


# In[44]:


dict_tit_desc = {}
for i in range(0, len(tit_desc_vocab)):
    dict_tit_desc[i] = tit_desc_vocab[i]

num = tfidf_features_titdesc.shape[0]
tfidf_weights_titdesc =[]

for i in range(0, num): 
    weight_dict = {}
    for word, value in zip(tit_desc_vocab, tfidf_features_titdesc.toarray()[i]): 
        if value > 0:
            weight_dict[word] = value
    tfidf_weights_titdesc.append(weight_dict)


# In[45]:


# TF-IDF Unweighted representation
unweighted_idf_titdes = gen_docVecs_unweighted(google_api, pd.Series(title_desc))
unweighted_idf_titdes.shape[0]


# In[46]:


# TF-IDF Weighted representation
weighted_idf_titdes = gen_docVecs_weighted(google_api, pd.Series(title_desc), tfidf_weights_titdesc)
weighted_idf_titdes.shape[0]


# In[47]:


# Model with title and description - Unweighted
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_idf_titdes, target, list(range(0,len(target))),test_size=0.33, random_state=0)
model = LogisticRegression(max_iter = 100, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[57]:


# # Model with title and description - Weighted
# X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(weighted_idf_titdes, target, list(range(0,len(target))),test_size=0.33, random_state=0)
# model = LogisticRegression(max_iter = 100, random_state=0)
# model.fit(X_train, y_train)
# model.score(X_test, y_test)


# From the above it is clear that adding information seems to lower the accuracy the model. 

# ## Summary
# Challenging tasks. Learnt many new stuffs and had faced many errors during this task, which were stepping stones to complete the assignment. 
