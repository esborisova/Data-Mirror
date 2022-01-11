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
    for i in data:
          try:
               for j in i['data']:
                    for key in j.keys():
                        if (key == 'post') and ('har skrevet') and ('tidslinje') not in i['title']:
                            posts.append(j[key])
          except KeyError:
              pass
    return posts 


def load_data(path: str, argument):
    new_path = os.path.join(path, argument)
    with open(new_path, encoding = 'utf-8') as f:
        dataset = json.load(f)
    return dataset



def create_txt(dataset: list, name, path):

    new_path = os.path.join(path, name)
    f = open(new_path, 'a')
    
    for item in dataset:
        without_line_breaks = item.replace("\n", " ")
        without_line_breaks = without_line_breaks.replace("\r", " ")
        lines = without_line_breaks + "\n"
        f.write(lines)
    f.close()

    

data = load_data('data/JSON_files/', str(sys.argv[1]))

all_posts = extract_posts(data)

print(len(all_posts))

create_txt(all_posts, name = 'all_data.txt', path = 'data/JSON_files/prepared_txt')
