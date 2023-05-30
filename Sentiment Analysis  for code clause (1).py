#!/usr/bin/env python
# coding: utf-8

# #  Sentiment Analysis in Python
# 
Sentiment Analysis in Python
In this notebook we will be doing some sentiment analysis in python using two different techniques:

1. VADER (Valence Aware Dictionary and sEntiment Reasoner) - Bag of words approach.
2. Roberta Pretrained Model from 🤗 .
3. Huggingface Pipeline .

# #  Step 0. Read in Data and NLTK Basic

# In[6]:


#   improting library .
#  Note that we applied ggplot2 styling to a histogram, but the statement plt. style. use('ggplot') can be used to apply 
#  ggplot2 styling to any plot in Matplotlib
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


import nltk


# In[4]:


df=pd.read_csv('Reviews.csv.zip')


# In[5]:


df


# In[7]:


# for shape of dataset rows and columns
df.shape


# In[8]:


df.isnull().sum()


# In[9]:


# for showing top 5 rows
df.head(5)


# In[10]:


# for bottom 5 rows
df.tail(5)


# In[12]:


# for text reviews of customers , In values using index values we can find any of  reviews of products
df['Text'].values[1]


# In[13]:


# we can find all info about dataset , here we can see memory usage columns names as well datatypes
df.info()


# In[15]:


# how many  each Scores accurs and  given by coustomers
df['Score'].value_counts()


# In[16]:


# sorts the Index values
df['Score'].value_counts().sort_index()


# In[17]:


# Sorts the Index values and then plots the graph 
df['Score'].value_counts().sort_index().plot(kind='bar',title=' Count of Reviews  by stars',figsize=(10,5))


# In[ ]:


ax= df['Score'].value_counts().sort_index().plot(kind='bar',title=' Count of Reviews  by stars',figsize=(10,5))


# In[30]:


# add labels and assigend as variable
ax=df['Score'].value_counts().sort_index() .plot(kind='bar',
      title=' Count of Reviews  by stars',
      figsize=(10,5))
ax.set_xlabel('Review by stars')
plt.show()


# In[28]:


# Here we can saw mostly we have 5 star rating , but also 1 star rating reviews too .
sns.countplot(x=df['Score'], data=df)
plt.title(' Count of Reviews  by stars',color='green')
plt.grid()
plt.show()


# In[53]:


# # BASIS NLTK(Natural language Toolkit)
# NLTK, or Natural Language Toolkit, is a Python package that we can use for NLP. A lot of the data that we could be
## analyzing is unstructured data and contains human-readable text

# lets take random 1 star example review

example=df['Text'][50]
example


# In[54]:


# Support for regular expressions (RE). 
#Support to pretty-print lists,tuples,dictionaries recursively.Very simple,but useful,especially in debugging data structures.clasess

import nltk, re, pprint


# In[55]:


import nltk
nltk.download('punkt')
  


# In[56]:


from nltk import word_tokenize


# In[58]:


nltk.word_tokenize(example)


# In[60]:


Token=nltk.word_tokenize(example)
Token[:10]


# In[62]:


# nltk.post-tag -> Use NLTK's currently recommended part of speech tagger to tag the given list of tokens.

import nltk
nltk.download('averaged_perceptron_tagger')
  


# In[73]:


# here 'NN' means noun ,and 'VBZ' means Verb
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.pos_tag(Token, tagset=None, lang='eng')


# In[76]:


tagged = nltk.pos_tag(Token)
tagged[:10]


# In[80]:


# now we take those tegs and put them into entities .
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.chunk.ne_chunk(tagged)


# In[81]:


entities=nltk.chunk.ne_chunk(tagged)
entities.pprint()


# # Step 1-> VADER Seniment Scoring, (Valence Aware Dictionary and sentiment                                                                                Reasoner)
# Step 1-> VADER Seniment Scoring (VADER-> (Valence Aware Dictionary and sentiment Reasoner) - Bag of words approach.

   We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text. It is an NLTK module that provides   sentiment scores based on the words used. It is a rule-based sentiment analyzer in which the terms are generally labeled as     per their semantic orientation as either positive or negative. This aprroch take all the words from sentence and it as values    postive, negative or neutral for each other words , combined up just as math equation for all the words and add up pos,         neg,ne   that statment is persent in this word . Keep in mind this approach does not account for the any relationship 
   between   words     which an human speech is very important .


This uses a "bag of words" approach:->
1. Stop words are removed (like -AND , THE etc.) which has not any postive or negative fellings .
2. each word is scored and combined to a total score. for sturecture of the sentence .
# In[83]:


import nltk
nltk.download('vader_lexicon')


# In[88]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[90]:


sia.polarity_scores('I am so happy!') # here we can see no  negative ,positive 0.682 - compound  sing 0.6468


# In[91]:


sia.polarity_scores('This is the worst thing ever.')  # here we can see no positive and - compound negative sing -0.6249


# In[92]:


sia.polarity_scores('I love this ice_cream most')


# In[94]:


example


# In[93]:


sia.polarity_scores(example)  # so here is overall negative Score , no positive review


# In[ ]:


df


# In[95]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[96]:


res 


# In[97]:


# convert it in pandas dataframe , bcz its easy to work with DataFrame , and .T use here to flip data in horizentaly
pd.DataFrame(res).T


# In[98]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[99]:


# Now we have sentiment score and metadata
vaders.head()


# In[100]:


# Plot VADER results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[103]:


sns.barplot(data=vaders, x='Score', y='pos')


# In[101]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# # Step 2. Roberta Pretrained Model
Step 2. Roberta Pretrained Model

* Use a model trained of a large corpus of data.

* Transformer model accounts for the words but also the context related to other words.Human language depends on context where    nagtive words also use for enjoyment and happiness , so this  ML model very use full there bcz they can pickup context

* This model is very popular in Deeplearing


RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.