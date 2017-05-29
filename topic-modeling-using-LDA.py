# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:06:26 2017

@author: Victor
"""
import json

doc_a=""
with open('yelp_academic_dataset_review-small-version.json') as json_data:
   for review in json_data:
       doc_a+=json.loads(review)['text'].lower()
      
#print(doc_a)    

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(doc_a)
print(tokens)

from stop_words import get_stop_words

# create English stop words list
en_stop = get_stop_words('en')

# remove stop words from tokens
stopped_tokens = [i for i in tokens if not i in en_stop]

print(stopped_tokens)

# To implement a Porter stemming algorithm, import the Porter Stemmer module from NLTK:
from nltk.stem.porter import PorterStemmer

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# p_stemmer requires all tokens to be type str. p_stemmer returns the string parameter 
# in stemmed form, so we need to loop through our stopped_tokens:

# stem token
stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

#print(stemmed_tokens)

# Let's construct the Document-Term Matrix (Dictionary)
from gensim import corpora, models

# list for tokenized documents in loop
# If we don't make "texts" list the corpora.Dictionary call will fail
texts = []
texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)

# The Dictionary() function traverses texts, assigning a unique integer id 
# to each unique token while also collecting word counts and relevant statistics.
# To see each token’s unique integer id, try print(dictionary.token2id).
#
#print(dictionary.token2id)

# Next, our dictionary must be converted into a BOW (bag-of-words):
corpus = [dictionary.doc2bow(text) for text in texts]

# The doc2bow() function converts dictionary into a bag-of-words. 
# The result, corpus, is a list of vectors equal to the number of documents.
# In each document vector is a series of tuples.
# As an example, print(corpus[0]) results in the following.
# The 1st tuple (0,2) says that 0 (brocolli) appears 2 times in the document (doc_a)
#print(corpus[0])

# Let's apply the LDA model
import gensim

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)

# Examining the results
# Our LDA model is now stored as ldamodel. 
# We can review our topics with the print_topic and print_topics methods:
#print(ldamodel.print_topics(num_topics=3, num_words=3))

# Lets' try now with 2 topics:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2, num_words=4))
