import langid

import codecs
import pandas as pd
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import spacy


import json
import sys
import pathlib
from pathlib import Path
import os

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


def only_da(dataset: list) -> list:

    langid.set_languages(['da', 'en'])  
    da = []

    for i in dataset:
        lang = langid.classify(i)
        if lang[0] != 'en':          
            da.append(i)

    return da


def remove_after_dot(argument):
    sep = '.'
    argument = argument.split(sep, 1)[0]

    return argument

def load_txt(path: str, argument):

    argument = remove_after_dot(argument)
    argument = argument + '.txt'

    new_path = os.path.join(path, argument)
    
    f = codecs.open(new_path, encoding = 'utf-8')
    data = f.read()
    data = data.splitlines()  

    return data


file = open("../stop_words.txt","r+")
stop_words = file.read().split()

data = load_txt('prepared_txt/', str(sys.argv[1]))

print(len(data))

da = only_da(data)

print(len(da))


nlp = spacy.load("da_core_news_sm")
stopwords_da = stopwords.words("danish")  
stopwords_da.extend(stop_words)

nlp1 = spacy.load("en_core_web_sm")
stopwords_en = stopwords.words("english")

titles = change_letter(da)

cleaned_titles = get_clean_text(titles, stops_da = stopwords_da, stops_en = stopwords_en)

remove_empty_strings = [string for string in cleaned_titles if string != ""]

df1 = pd.DataFrame(remove_empty_strings , columns=['cleaned_titles'])

tokenized_titles= tokenize_text(remove_empty_strings)

lemmatized_titles = lemmatize_posts(tokenized_titles)

df1['tokenized_titles'] = tokenized_titles
df1['lemmatized_titles'] = lemmatized_titles

name = remove_after_dot((sys.argv[1]))
name = name + '.pkl'
path = 'prepared_pkl/' + name

df1.to_pickle(path)