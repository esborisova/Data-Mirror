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
import pathlib
from pathlib import Path
import os
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from functions import *



nlp = spacy.load("da_core_news_sm")
stops_path = '../stop_words.txt'
file_path = ' '

file = open(stops_path,"r+")
stops = file.read().split()

data = load_json(file_path)

titles = extract_title(data)

titles = change_letter(titles)

cleaned_titles = clean_text(titles)

no_empty_strings = [string for string in cleaned_titles if string != ""]

tokenized_titles= tokenize(no_empty_strings)

lemmatized_titles = lemmatize(tokenized_titles, nlp = nlp)

lemmatized_titles = change_char(lemmatized_titles)

no_stops = remove_stops(lemmatized_titles, stops)

no_stops = [l for l in no_stops if len(l) != 0]

no_stops = rem_single_char(no_stops)

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


best_result_index = coherence_values.index(max(coherence_values))


ldamodel = LdaModel(corpus = corpus, 
                    num_topics = x[best_result_index], 
                    id2word = dictionary,
                    update_every = 1,
                    passes = 10,
                    per_word_topics = True)


model = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)

save_model(model, ' ', ' ')



