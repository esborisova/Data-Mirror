import pandas as pd
import sys
import pandas as pd
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import spacy
import pathlib
from pathlib import Path
import os
import codecs

import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim_models
import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.models import CoherenceModel



# Remove .json in the file name to create and use further a new name
def remove_after_dot(argument):
    sep = '.'
    argument = argument.split(sep, 1)[0]

    return argument



# Load txt file
def load_txt(path: str, argument):
    argument = remove_after_dot(argument)
    argument = argument + '.txt'

    new_path = os.path.join(path, argument)
    f = codecs.open(new_path, encoding = 'utf-8')
    data = f.read()
    data = data.splitlines()  

    return data


# Change Danish special letters
def change_letter(dataset: list) -> list:
    new_dataset = [re.sub('ø', 'oe', text) for text in dataset]
    new_dataset = [re.sub('æ', 'ae', text) for text in new_dataset]
    new_dataset = [re.sub('å', 'aa', text) for text in new_dataset]
    new_dataset = [re.sub('Ø', 'oe', text) for text in new_dataset]
    new_dataset = [re.sub('Æ', 'ae', text) for text in new_dataset]
    new_dataset = [re.sub('Å', 'aa', text) for text in new_dataset]
    new_dataset = [re.sub('ü', 'ue', text) for text in new_dataset]
    new_dataset = [re.sub('Ü', 'ue', text) for text in new_dataset]
    new_dataset = [re.sub('ä', 'ae', text) for text in new_dataset]
    new_dataset = [re.sub('Ä', 'ae', text) for text in new_dataset]
    new_dataset = [re.sub('ö', 'oe', text) for text in new_dataset]
    new_dataset = [re.sub('Ö', 'oe', text) for text in new_dataset]
    
    return new_dataset



def change(data):
    for i in range (len(data)):
        data[i] = change_letter(data[i] )
    return data



# Remove URLs, special charecters, double spaces, spaces in the begining and end of a string, lowercase 
def get_clean_text(dataset: list) -> list:

    no_urls = [re.sub(r"http\S+", "", text) for text in dataset] 
    only_letters = [re.sub(r"(#[A-Za-z]+)|(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", ' ' , text) for text in no_urls] 
    only_letters = [text.replace('\n', ' ') for text in only_letters]
    lowercased_str = [text.lower() for text in only_letters] 
    cleaned_text = [re.sub(' +', ' ', text) for text in lowercased_str]  
    final_text = [text.strip() for text in cleaned_text]

    return final_text

# Tokenize posts 
def tokenize_text(data):
    tokens = [word_tokenize(text) for text in data] 
    return tokens

# Lemmatize tokens 
def lemmatize_text(sent):
    lemmas = [x.lemma_ for x in nlp(sent)]
    return lemmas



def lemmatize_posts(tokenized_data: list):
    lemmas = []
    for post in tokenized_data:
        lemma = [lemmatize_text(x) for x in post]
        lemmas.append([item for sublist in lemma for item in sublist])
    return lemmas


# Remove tokens/strings containing only 1 letter istead of a word 
def remove_1_letter(data: list):
    for i in data:
        for j in i:
            if len(j)<2:
                i.remove(j)
    return data


# Remove stop words
def remove_stops(data: list,
                 stopwords: list):

    no_stopwords = []

    for i in data:
        tmp_list = []
        for j in i:
            if j not in stopwords:
                tmp_list.append(j)
        no_stopwords.append(tmp_list)

    return no_stopwords



# Compute c_v coherence for various number of topics
def compute_coherence_values(dictionary, 
                             corpus, 
                             texts, 
                             max_topics, 
                             min_topics, 
                             step):

    coherence_values = []
    model_list = []
    
    for num_topics in range(min_topics, max_topics, step):

        model = LdaModel(corpus = corpus, 
                         num_topics = num_topics, 
                         id2word = dictionary,
                         update_every = 1,
                         passes = 10,
                         per_word_topics = True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, 
                                        texts = texts, 
                                        dictionary = dictionary, 
                                        coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



# Load model for Danish text preprocessing (it is used for lemmatization) 
nlp = spacy.load("da_core_news_sm")


# Load stop words list
file = open("../stop_words_extended.txt","r+")
stops = file.read().split()

data = load_txt('prepared_txt/', str(sys.argv[1]))

titles = change_letter(data)

cleaned_titles = get_clean_text(titles)


# Check for and remove empty strings again 
remove_empty_strings = [string for string in cleaned_titles if string != ""]

tokenized_titles= tokenize_text(remove_empty_strings)

tokenized = remove_1_letter(tokenized_titles)

lemmatized_titles = lemmatize_posts(tokenized)

# For some reason we again have words with special letters 
lemmatized_titles = change(lemmatized_titles)


# Prepare stop words. We can do that in advance once it is final
stops = " ".join(stops)
stops = nlp(stops)
my_stop_words = [t.lemma_ for t in stops]
my_stop_words = change_letter(my_stop_words)

no_stops = remove_stops(lemmatized_titles, my_stop_words)

# Check for empty lists and remove them
remove_empty_list = [l for l in no_stops if len(l) != 0]

#Check for single letters and remove them again 
no_1_letter = remove_1_letter(remove_empty_list)


# Prepare the dictionary of bigrams for the model
bigram = gensim.models.Phrases(no_1_letter)
titles_bigrams = [bigram[line] for line in no_1_letter]

dictionary = Dictionary(titles_bigrams)
corpus = [dictionary.doc2bow(text) for text in titles_bigrams]



# We can think about the script which checks the dataset size 
# and apply the max number of topics para depending on that
# F.ex.: if the size of D < 10000, max_topics = 10,  if 10000 < D <20000,  max_topics = 15, etc.

min_topics = 2
max_topics = 15
step = 2
x = range(min_topics, max_topics, step)

# Compute the c_v score
model_list, coherence_values = compute_coherence_values(dictionary = dictionary, 
                                                        corpus = corpus, 
                                                        texts = titles_bigrams, 
                                                        min_topics = min_topics, 
                                                        max_topics = max_topics, 
                                                        step = step)


# Get the c_v score and put it into the num_topics param of the model
best_result_index = coherence_values.index(max(coherence_values))
print(x[best_result_index])

ldamodel = LdaModel(corpus = corpus, 
                    num_topics = x[best_result_index], 
                    id2word = dictionary,
                    update_every = 1,
                    passes = 10,
                    per_word_topics = True)

model =pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)


new_name = remove_after_dot((sys.argv[1]))
new_name = new_name + '.html'
path = 'models/' + new_name

pyLDAvis.save_html(model, path)
