#pip install nltk
#pip install spacy
#python -m spacy download da_core_news_sm

#pip install gensim
#pip install pyLDAvis

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


import pathlib
from pathlib import Path
import os

import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim_models
import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation




def extract_posts(dataset: pd.DataFrame) -> list:
    posts = []
    for i in dataset:
        try:
            if ('har skrevet' in i['title']) and ('tidslinje' in i['title']):
                continue
        except KeyError:
            pass
        try:
            for j in i['data']:
                for key in j.keys():
                    if (key == 'post'):
                        posts.append(j[key])
        except KeyError:
            pass
    return posts



def change_letter(dataset: pd.DataFrame) -> list:
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


def get_clean_text(dataset: list, stops_da, stops_en) -> list:
    no_urls = [re.sub(r"http\S+", "", text) for text in dataset] 
    only_letters = [re.sub(r"(#[A-Za-z]+)|(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", ' ' , text) for text in no_urls] 
    only_letters = [text.replace('\n', ' ') for text in only_letters]
    lowercased_str = [text.lower().split() for text in only_letters] 
    no_stopwords = [[w for w in text if not w in stops_da] for text in lowercased_str] 
    no_stopwords = [[w for w in text if not w in stops_en] for text in no_stopwords] 
    cleaned_text = [" ".join(text) for text in no_stopwords] 
    return cleaned_text


def tokenize_text(data):
    tokens = [word_tokenize(text) for text in data] 
    return tokens


def lemmatize_text(sent):
    lemmas = [x.lemma_ for x in nlp(sent)]
    return lemmas

def lemmatize_posts(tokenized_data: list):
    lemmas = []
    for post in tokenized_data:
        lemma = [lemmatize_text(x) for x in post]
        lemmas.append([item for sublist in lemma for item in sublist])
    return lemmas


def load_data(path: str, argument):
    new_path = os.path.join(path, argument)
    with open(new_path, encoding = 'utf-8') as f:
        dataset = json.load(f)
    return dataset


def remove_after_dot(argument):
    sep = '.'
    argument = argument.split(sep, 1)[0]
    return argument



def create_txt(dataset: list, argument, path):
    argument = remove_after_dot(argument)
    argument = argument + '.txt'

    new_path = os.path.join(path, argument)
    f = open(new_path, 'w')
    
    for item in dataset:
        without_line_breaks = item.replace("\n", " ")
        without_line_breaks = without_line_breaks.replace("\r", " ")
        lines = without_line_breaks + "\n"
        f.write(lines)
    f.close()

    

def decode(path: str, argument) -> list:
    argument = remove_after_dot(argument)
    argument = argument + '.txt'

    new_path = os.path.join(path, argument)
    f = codecs.open(new_path, errors = 'ignore', encoding = 'utf-8')
    data = f.read()
    data = data.encode('latin1').decode('utf8')
    data = data.splitlines()  
    return data    


def remove_mentions(dataset: list):
    new_dataset = []
    for i in dataset:
        if i.startswith('@'):
          i = i.replace(i, "")
        new_dataset.append(i)
    return new_dataset

#data = load_data('data/JSON_files/your_posts_1AF.json')

data = load_data('data/JSON_files/', str(sys.argv[1]))

all_posts = extract_posts(data)

print(len(all_posts))

create_txt(all_posts, str(sys.argv[1]), 'data/JSON_files/output_txt_updated/')

decoded_data = decode('data/JSON_files/output_txt_updated/', str(sys.argv[1]))

#print(len(decoded_data))

new_data = remove_mentions(decoded_data)

#print(len(new_data))

without_empty_strings = [string for string in new_data if string != ""]

print(len(without_empty_strings))

df = pd.DataFrame(without_empty_strings, columns=['row_posts'])

nlp = spacy.load("da_core_news_sm")
stopwords_da = stopwords.words("danish")  
stopwords_da.extend(['paa', 'saa', 'vaere',  'rt', 'ogsaa', 'faa', 'faar', 'nok', 'mt', 'gt'])

nlp1 = spacy.load("en_core_web_sm")
stopwords_en = stopwords.words("english")

fb_posts = change_letter(df['row_posts'])

cleaned_posts = get_clean_text(fb_posts, stops_da = stopwords_da, stops_en = stopwords_en)

remove_empty_strings = [string for string in cleaned_posts if string != ""]

tokenized_posts = tokenize_text(remove_empty_strings)

lemmatized_posts = lemmatize_posts(tokenized_posts)


df1 = pd.DataFrame(remove_empty_strings , columns=['cleaned_posts'])

df1['tokenized_posts'] = tokenized_posts
df1['lemmatized_posts'] = lemmatized_posts


bigram = gensim.models.Phrases(df1['lemmatized_posts'])
posts_bigrams = [bigram[line] for line in df1['lemmatized_posts']]

dictionary = Dictionary(posts_bigrams)
corpus = [dictionary.doc2bow(text) for text in posts_bigrams]

ldamodel = LdaModel(corpus = corpus, 
                    num_topics = 5, 
                    id2word = dictionary,
                    update_every = 1,
                    passes = 10,
                    per_word_topics = True)

#pyLDAvis.enable_notebook()
model =pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)

argument = remove_after_dot(str(sys.argv[1]))
argument = argument + '.html'
model_path = os.path.join('data/JSON_files/models/', argument)
pyLDAvis.save_html(model, model_path)

print(str(sys.argv[1]))

