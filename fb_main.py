import json
import pandas as pd
import spacy
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import codecs
import sys
import datetime

import json
import sys
import pathlib
from pathlib import Path
import os


import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim_models
import pyLDAvis.sklearn


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from functions import *

nlp = spacy.load("da_core_news_sm")

# Load stop words list

stops_path = 'stops_final.txt'
file_path = 'data/fb/JSON_files/your_posts_1_AG.json'

file = open(stops_path,"r+")
stops = file.read().split()


data = load_json(file_path)

# for generic model
#date = '1/1/2020' 

extracted_posts = extract_posts(data)

save_txt(extracted_posts, 'data/fb/prepared_txt', 'your_posts_1_AG2.txt')

decoded_data = decode('data/fb/prepared_txt/your_posts_1_AG2.txt')

no_mentions = remove_mentions(decoded_data)

# Remove any strings which became empty as a result of mentions filtering 
no_empty_str = [string for string in  no_mentions if string != ""]

no_empty_str = change_letter(no_empty_str)

cleaned_data = clean_text(no_empty_str)

# Check for and remove empty strings again 
cleaned_data = [string for string in cleaned_data if string != ""]

tokenized_data = tokenize(cleaned_data)

lemmatized_data = lemmatize(tokenized_data, nlp = nlp)

# After lemmatization we again have words with special letters 
lemmatized_data = change_char(lemmatized_data)

no_stops = remove_stops(lemmatized_data, stops)

# Check for empty lists and remove them
no_stops = [l for l in no_stops if len(l) != 0]

no_stops = rem_single_char(no_stops)

# Prepare the dictionary of bigrams for the model
bigram = gensim.models.Phrases(no_stops)

dictionary = Dictionary(no_stops)
corpus = [dictionary.doc2bow(text) for text in no_stops]

max_topics = set_max_topics(no_stops)
min_topics = 2
step = 2
x = range(min_topics, max_topics, step)

coherence_values = compute_c_v(dictionary = dictionary, 
                               corpus = corpus, 
                               texts = no_stops, 
                               min_topics = min_topics, 
                               max_topics = max_topics, 
                               step = step)


# Get the c_v score and put it into the num_topics param of the model
best_result_index = coherence_values.index(max(coherence_values))


ldamodel = LdaModel(corpus = corpus, 
                    num_topics = x[best_result_index], 
                    id2word = dictionary,
                    update_every = 1,
                    passes = 10,
                    per_word_topics = True)


model = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)

save_model(model, 'models', 'your_posts_1_AG.html')