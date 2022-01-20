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


# Load json
def load_data(path: str, argument):

    new_path = os.path.join(path, argument)
    
    with open(new_path, encoding = 'utf-8') as f:
        dataset = json.load(f)
    return dataset


# Extract posts from json file
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


# Remove .json from the file name to create and use further a new name
def remove_after_dot(argument):
    sep = '.'
    argument = argument.split(sep, 1)[0]
    return argument


# Save extracted posts (from each json file) to a seperate txt file 
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



data = load_data('JSON_files/', str(sys.argv[1]))

all_posts = extract_posts(data)

#print(len(all_posts))

create_txt(all_posts, str(sys.argv[1]), 'prepared_txt/')