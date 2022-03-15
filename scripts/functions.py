import sys
import pathlib
from pathlib import Path
import os
import os
import datetime as dt
import time
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





def load_json(path: str) -> list:
    """Loads json file and saves in a list

    Args:
       path (str): Path to a file

    Returns:
          dataset (list): The list of strings
    """
    
    with open(path, encoding = 'utf-8') as f:
        dataset = json.load(f)
    return dataset



def extract_posts(dataset: list,
                  date: str = None) -> list:
    """Extracts Facebook posts from a json file

    Args:

        dataset (list): A list with json formated data

        date (str): An optional argument. The date from which to collect the posts. The date should have '%d/%m/%y' format. 

    Returns:
          posts (list): The list of strings (posts text) 
    """
        
    text = []

    for i in dataset:
        if date != None:
            try:
                if ('har skrevet' in i['title']) and ('tidslinje' in i['title']) and (i['timestamp']) and (datetime.datetime.fromtimestamp(i['timestamp']).strftime('%d/%m/%y') >= date):
                    continue
            except KeyError:
                pass
        else: 
            try:
                if ('har skrevet' in i['title']) and ('tidslinje' in i['title']) and (i['timestamp']):
                    continue
            except KeyError:
                pass
            
        try:
            for j in i['data']:
                for key in j.keys():
                    if (key == 'post'):
                        text.append(j[key])
        except KeyError:
            pass
        
    return text



def extract_title(dataset: list) -> list:
    """Extracts Google browse titles from a json file

    Args:
        dataset (list): A list with json formated data

        date (str): An optional argument. The date from which to collect the posts. The date should have '%d/%m/%y' format. 

    Returns:
          titles (list): The list of strings (title's text) 
    """

    titles = []

    for i in dataset['Browser History']:
            try:
                 titles.append(i['title'])
            except KeyError:
                    pass
    return titles



#def save_txt(dataset: list, 
#             folder_name: str, 
#             file_name: str):
#    """Creates a folder and saves data into a txt file 
#
#    Args:
#        dataset (list): A list of strings 
#        
#        folder name (str): The name of a folder to create
#
#        file_name (str): The name of a txt file with data
#    """
#    
#    current_dir = pathlib.Path().resolve()
#    path = os.path.join(current_dir, folder_name)
#
#    if not os.path.exists(folder_name):
#        os.mkdir(path)
#
#    file_path = os.path.join(path, file_name) 
#    
#    if os.path.exists(file_path):
#        pass
#
#    f = open(file_path, 'w')
#    
#    for item in dataset:
#        without_line_breaks = item.replace("\n", " ")
#        without_line_breaks = without_line_breaks.replace("\r", " ")
#        lines = without_line_breaks + "\n"
#        f.write(lines)
#    f.close()



def save_txt(dataset: list,  
              file_path: str):
    f = open(file_path, 'w')
    
    for item in dataset:
        without_line_breaks = item.replace("\n", " ")
        without_line_breaks = without_line_breaks.replace("\r", " ")
        lines = without_line_breaks + "\n"
        f.write(lines)
    f.close()



def decode(path: str) -> list:

    f = codecs.open(path, errors = 'ignore', encoding = 'utf-8')
    data = f.read()
    data = data.encode('latin1').decode('utf8')
    data = data.splitlines()  

    return data



def remove_mentions(dataset: list) -> list:

    new_dataset = []

    for i in dataset:
        if i.startswith('@'):
          i = i.replace(i, "")
        new_dataset.append(i)

    return new_dataset



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



def clean_text(dataset: list) -> list:
   
    no_urls = [re.sub(r"http\S+", "", text) for text in dataset] 
    no_special_ch = [re.sub(r"(#[A-Za-z]+)|(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", ' ' , text) for text in no_urls] 
    no_special_ch  = [text.replace('\n', ' ') for text in no_special_ch ]
    lowercased_str = [text.lower() for text in no_special_ch] 
    cleaned_text = [re.sub(' +', ' ', text) for text in lowercased_str]  
    cleaned_text = [text.strip() for text in cleaned_text]

    return cleaned_text



def tokenize(dataset: list) -> list:
    
    tokens = [word_tokenize(text) for text in dataset] 

    return tokens



def lemmatize_str(text: list, nlp) -> list:

    lemmas = [x.lemma_ for x in nlp(text)]

    return lemmas



def lemmatize(dataset: list, nlp) -> list:
  
    lemmas = []

    for item in dataset:
        lemma = [lemmatize_str(x, nlp = nlp) for x in item]
        lemmas.append([item for sublist in lemma for item in sublist])

    return lemmas



def remove_stops(dataset: list,
                 stopwords: list) -> list:

    no_stopwords = []

    for i in dataset:
        tmp_list = []
        for j in i:
            if j not in stopwords:
                tmp_list.append(j)
        no_stopwords.append(tmp_list)

    return no_stopwords



def remove_empty_str(data: list):
    result = [string for string in data if string != ""]
    return result



def change_char(dataset: list) -> list:

    for i in range (len(dataset)):
        dataset[i] = change_letter(dataset[i] )
    return dataset



def rem_single_char(dataset: list) -> list:
    
    for i in dataset:
        for j in i:
            if len(j)<2:
                i.remove(j)
    return dataset


def dict_settings(data: list):
   
    bigram = gensim.models.Phrases(data)
    dictionary = Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]

    return bigram, dictionary, corpus



def topic_settings(data: list):
    
    max_topics = set_max_topics(data)
    min_topics = 2
    step = 2
    x = range(min_topics, max_topics, step)

    return x, min_topics, max_topics, step



def compute_c_v(dictionary, 
                corpus: list, 
                texts: list, 
                min_topics: int, 
                max_topics: int, 
                step: int) -> list:
    """Computes c_v coherence score 

    Args:
       dictionary: A list of tuples (word_id, word_frequency)

       corpus (list): A document-term matrix 

       texts (list): A list of lists with lemmatized strings (tokens)

       min_topics (int): The minimum number of topics to identify

       max_topics (int): The maximum number of topics to identify
              
       step (int): Step with which to iterete across the topics

    Returns:
          coherence_values (list): The list with c_v scores for each number of topics          
    """

    coherence_values = []
    
    for num_topics in range(min_topics, max_topics, step):

        model = LdaModel(corpus = corpus, 
                         num_topics = num_topics, 
                         id2word = dictionary,
                         update_every = 1,
                         passes = 10,
                         per_word_topics = True)

        coherencemodel = CoherenceModel(model = model, 
                                        texts = texts, 
                                        dictionary = dictionary, 
                                        coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values



def set_max_topics(dataset: list) -> int:
    """Defines the maximum number of topics for the c_v score

    Args:
        dataset: A list of strings or lists

    Returns:
          max_topics (int): The number of topics
    """

    size = len(dataset)

    if size < 10000:
        max_topics = 10
        
    else:
        max_topics = 15

    return max_topics



def save_model(model, 
               folder_name: str, 
               file_name: str):
    """Creates a folder and saves topic model in it

    Args:
       model: pyLDAvis gensim model

       folder_name (str): the name of the folder to be created

       file_name (str): The name of the html file with the model  
    """
  
    current_dir = pathlib.Path().resolve()
    path = os.path.join(current_dir, folder_name)

    if not os.path.exists(folder_name):
        os.mkdir(path)

    file_path = os.path.join(path, file_name) 
    
    #if os.path.exists(file_path):
    #    pass

    pyLDAvis.save_html(model, file_path)


def clean_folder(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            os.unlink(file_path)
    
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_time(file_path: str):

    st = os.stat(file_path)    
    atime = dt.datetime.fromtimestamp(st.st_atime)
    now = dt.datetime.now()
    ago = now-dt.timedelta(minutes = 1)

    return atime, ago

def prepare_input(data):
    
    bigram = gensim.models.Phrases(data)
    dictionary = Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
                        
    return bigram, dictionary, corpus


def slice_data(data):
                            
    n = 20

    if len(data) > n:
        data_generic = data[-n:]
    else:
        data_generic = data

    return data_generic

def txt_append(path, data):

    file_object = open(path, 'a')
    file_object.write(str(data)[1:-1])
    file_object.write(',')
    file_object.close()
