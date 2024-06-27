#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
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
# * load_files
# * RegexpTokenizer
# * sent_tokenize
# * chain
# 
# ## Introduction
# In this task, the basic text pre-processing is performed on the given dataset. Following steps were performed to get desired output:
# - Tokenization
# - Convert tokens to lowercase
# - Remove words less than 2 characters
# - Remove all the stop words
# - Remove words that appear once in document based on term frequency
# - Remove the top 50 most frequent words based on document frequency
# - Saving all the information to separate text files
# - Finally, generate the vocabulary file with cleaned job descriptions

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
from sklearn.datasets import load_files
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
import nltk
nltk.download('punkt')
from nltk.probability import *


# ### 1.1 Examining and loading data
# 
# Before doing any pre-processing, the data has to be loaded into a proper format. To load the data, the data folder has to be explored. Inside the `data` folder, we have a sub-directories named `Accounting_Finance`, `Engineering`, `Healthcare_Nursing`, and `Sales` which accounts for a total of 776 files inside those.
# 
# To load the data, `load_files` from `sklearn.datasets` is used.
# 
# The loaded `job_data` is then a dictionary, with the following attributes:
# * `data` - a list of text reviews
# * `target` - the corresponding label of the text reviews (integer index)
# * `target_names` - the names of target classes.
# * `filenames` - the filenames holding the dataset.
# 
# Once the data is loaded, it is then ready for pre-processing which is the next step where we extract each information to separate text files and use them for later purposes.

# In[2]:


# Code to inspect the provided data file...
job_data = load_files(r"data")
job_data.keys()


# In[3]:


job_data['filenames']


# In[4]:


job_data['data']


# In[5]:


job_data['target_names'] 


# In[6]:


job_data['target'] # this means value 0 is 'Accounting_Finance', 1 is 'Engineering', 2 is 'Healthcare_Nursing', 3 is 'Sales'


# In[7]:


# test whether it matches, just in case
emp = 0
job_data['filenames'][emp], job_data['target'][emp]


# In[8]:


job_desc, job_category = job_data.data, job_data.target  


# In[9]:


job_desc[emp]


# In[10]:


job_category[emp]


# ### 1.2 Pre-processing data
# In this step, the required text pre-processing are performed.
# 
# Once successfully the data is loaded, pre-processing begins. In the following sub-tasks, the following basic text pre-processing steps are tackled one by one:
# 
# * Word Tokenization
# * Convert tokens to lowercase
# * Removing Tokens with less than 2 characters
# * Removing Stop words
# * Remove words that appear once in document based on term frequency
# * Remove the top 50 most frequent words based on document frequency

# ###### 1. Extract information from each job advertisement

# ###### 1.1 Extracting job description

# In[11]:


job_description = []

for job in job_desc:
    # decode to string
    string = job.decode()
    
    # find the index where description starts
    result = string[string.find('Description: '):]
    
    # append to the list
    # result[13:] - because we need values after 'Description: ' which is of length 13
    job_description.append(result[13:])
print(len(job_description))


# In[12]:


# Storing all the job descriptions to a text file 
with open("job_descriptions.txt", "w", encoding="utf-8") as f:
    for i in range(0,len(job_description)):
        f.write(job_description[i] + "\n")


# ###### 1.2 Extracting job web index

# In[13]:


job_web_indices = []

for job in job_desc:
    
    # decode to string
    string = job.decode('utf-8')
    
    # find the index where webindex starts
    result = string.find('Webindex: ')
    
    # append to the list
    # result+10 - because we need values after 'Webindex: ' which is of length 10
    # result+18 - because for the 8 digits that comes after 'Webindex: '
    job_web_indices.append(string[result+10: (result+18)])
print(len(job_web_indices))


# In[14]:


# Storing all the job web index to a text file 
with open("web_indices.txt", "w") as f:
    for i in range(0, len(job_web_indices)): 
        f.write(job_web_indices[i] + "\n")


# ###### 1.3 Extracting job titles

# In[15]:


job_titles = []

for job in job_desc:
    
    # decode to string
    string = job.decode('utf-8')
    
    # find the index where Title starts
    start = string.find('Title: ')
    
    # find the index where Webindices starts which is the end of Title
    end = string.find('Webindex: ')
    
    # append to the list
    # start+7 - because we need values after 'Title: '
    # end-1 - because we need to ignore the space at the end
    job_titles.append(string[start+7 : end-1])

print(len(job_titles))


# In[16]:


# Storing all the job titles to a text file 
with open("job_titles.txt", "w") as f:
    for i in range(0, len(job_titles)):
        f.write(job_titles[i] + "\n")


# ###### 1.4 Extracting companies

# In[17]:


company_list = []

for job in job_desc:
    
    # decode to string
    string = job.decode('utf-8')
    
    # find the index where Company starts
    start = string.find('Company: ')
    
    # find the index where description starts which is the end of Company
    end = string.find('Description: ')
    
    # append to the list 
    # start+9 - because we need values after 'Company: ' which is of length 9
    # end-1 - because we need to ignore the space at the end
    company_list.append(string[start+9 : end-1])

print(len(company_list))


# In[18]:


# Storing all the company names to a text file 
with open("companies.txt", "w") as f:
    for i in range(0, len(company_list)):
        f.write(company_list[i] + "\n")


# ###### 2. Tokenize each job advertisement description - r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

# In[19]:


def tokenizeDesc(raw_jobdata):
    """
        This function segments the raw job data into sentences and tokenize each sentences 
        and convert the review to a list of tokens.
    """
    # convert all words to lowercase
    nl_jobdata = raw_jobdata.lower()
    
    # segment into sentences
    sentences = sent_tokenize(nl_jobdata)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_jobdata = list(chain.from_iterable(token_lists))
    return tokenised_jobdata


# In[20]:


# Generate a list of tokenized articles
tk_desc = [tokenizeDesc(desc) for desc in job_description]


# In[21]:


# check if the descrpition is tokenized
print("Raw JD:\n",job_description[0],'\n')
print("Tokenized JD:\n",tk_desc[0])


# ###### 3. Covert all tokens to lowercase

# In[22]:


tk_desc_lower = []
for i in range(0, len(tk_desc)):
    # convert each token to lower case and append it to the list
    tk_desc_lower.append([tk.lower() for tk in tk_desc[i]])


# ###### 4. Remove words with length less than 2

# In[23]:


words_g2 = [[tk for tk in low_tk if len(tk) >= 2] for low_tk in tk_desc_lower]


# ###### 5. Remove stopwords

# In[24]:


stopwords_list = []

with open('./stopwords_en.txt') as f:
    stopwords_list = f.read().splitlines()


# In[25]:


rem_stopwords = [[w for w in word if w not in stopwords_list] for word in words_g2]


# ###### 6. Remove the words that appear only once in the document collection, based on term frequency

# In[26]:


words = list(chain.from_iterable(rem_stopwords))

# compute term frequency for each unique word/type
term_freq = FreqDist(words)
lessFreqWords = set(term_freq.hapaxes())


# In[27]:


def removeLessFreqWords(words):
    '''
        This function is to remove the words that appear only once in document based on term frequency
    '''
    return [w for w in words if w not in lessFreqWords]

tk_removeLessTermFreq = [removeLessFreqWords(word) for word in rem_stopwords]


# ###### 7. Remove the top 50 most frequent words based on document frequency

# In[28]:


words_2 = list(chain.from_iterable([set(tk) for tk in tk_removeLessTermFreq]))

# find words that appear most commonly across documents
doc_freq = FreqDist(words_2)  
doc_freq_sorted = sorted(list(doc_freq.most_common(50)))


# In[29]:


# Creating a list to append the top 50 words
doc_freq_words = []
for i,j in doc_freq_sorted:
    doc_freq_words.append(i)


# In[30]:


def removeTop50(words):
    '''
        This function is to remove top 50 frequent words based on document frequency
    '''
    return [word for word in words if word not in doc_freq_words]

tk_removeMostDocumentFreq = [removeTop50(words) for words in tk_removeLessTermFreq]


# ## Saving required outputs
# 
# In here, the following steps are performed:
# 
# - Saving all the information to separate text files
# - Build the vocabulary file with cleaned job descriptions

# ###### 8. Save all job advertisement text and information in txt file(s)

# In[31]:


# Storing the job description and job category to new text files
def save_desc(descFilename,tk_removeMostDocumentFreq):
    out_file = open(descFilename, 'w') # creates a txt file and open to save the reviews
    string = "\n".join([" ".join(desc) for desc in tk_removeMostDocumentFreq])
    out_file.write(string)
    out_file.close() # close the file

def save_label(labelFilename,job_category):
    out_file = open(labelFilename, 'w') # creates a txt file and open to save sentiments
    string = "\n".join([str(s) for s in job_category])
    out_file.write(string)
    out_file.close() # close the file 
    
save_desc('job_desc.txt',tk_removeMostDocumentFreq)
save_label('job_category_target.txt',job_category)


# ###### 9. Build a vocabulary of the cleaned job advertisement descriptions, save it in a txt file

# In[32]:


words = list(chain.from_iterable([set(tk) for tk in tk_removeMostDocumentFreq]))
vocab = sorted(list(set(words)))

# Write the vocab to a new text file
vocab_file = open("vocab.txt", 'w')
for idx in range(0, len(vocab)):
    # write each index and word, index starts from 0
    vocab_file.write(f"{vocab[idx]}:{idx}\n")
vocab_file.close()


# ## Summary
# 
# Being very new to NLP, I am glad that I had a chance to work on it. Moreover, I am very excited that I had reached the end of task 1 and built the vocabulary of cleaned job advertisement descriptions as **vocab.txt**. Adding on, learnt more about tokenizations, term and document frequencies and how to write each detailed information in separate text files.

# In[ ]:




