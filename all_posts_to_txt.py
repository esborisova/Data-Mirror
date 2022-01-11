import json
import pandas as pd
import re
import codecs
import sys


import pathlib
from pathlib import Path
import os


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

create_txt(all_posts, name = 'all_data_updated.txt', path = 'data/JSON_files/output_txt_updated')
