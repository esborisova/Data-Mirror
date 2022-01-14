import json
import pandas as pd
import spacy
import re
import codecs
import sys
import pathlib
from pathlib import Path
import os


def load_data(path: str, argument):
    new_path = os.path.join(path, argument)
    with open(new_path, encoding = 'utf-8') as f:
        dataset = json.load(f)
    return dataset



def extract_title(dataset: pd.DataFrame) -> list:
    titles = []
    for i in dataset['Browser History']:
            try:
                 titles.append(i['title'])
            except KeyError:
                    pass
    return titles



def create_txt(dataset: list, name, path):

    new_path = os.path.join(path, name)
    f = open(new_path, 'a')
    
    for item in dataset:
        without_line_breaks = item.replace("\n", " ")
        without_line_breaks = without_line_breaks.replace("\r", " ")
        lines = without_line_breaks + "\n"
        f.write(lines)
    f.close()


data = load_data('JSON_files/', str(sys.argv[1]))

all_titles = extract_title(data)

#my_file = open("sum", "a")
#my_file.write(str(len(all_titles)))
#my_file.write("\n")
#my_file.close()

create_txt(all_titles, name = 'all_titles.txt', path = 'prepared_txt/')