"""Pipeline for bulding a topic model based on Google data."""

import sys
import pathlib
from pathlib import Path
import os
from load_save import load_json, save_model, not_processed
from preprocess import (
    change_letter,
    clean_text,
    tokenize,
    lemmatize,
    change_char,
    remove_stops,
    rem_single_char,
    remove_empty_str,
)
from extrectors import extract_title
from topic_modelling import prepare_input, set_max_topics, compute_c_v
from utils import get_time, check_data_type, slice_data, txt_append
import datetime as dt
import time
import json
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
import codecs
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load("da_core_news_sm")
stops_path = "../../stops/G_stop_words.txt"

while True:
    for file in os.listdir("../json/"):
        if "G" in file:

            file_path = os.path.join("../json/", file)
            atime, ago = get_time(file_path)

            if atime <= ago:

                stops = open(stops_path, "r+")
                stops = stops.read().split()

                file_name = file.split("_")[-1].split(".")[0]

                try:
                    data = load_json(file_path)
                    data_type = check_data_type(data)

                    if data_type == True:
                        titles = extract_title(data)

                        if len(titles) == 0:
                            not_processed(
                                folder_path="../json/",
                                file_name=file,
                                folder_to_save="../not_processed/",
                            )
                        else:
                            titles = change_letter(titles)
                            cleaned_titles = clean_text(titles)
                            no_empty_strings = remove_empty_str(cleaned_titles)
                            tokenized_titles = tokenize(no_empty_strings)
                            lemmatized_titles = lemmatize(tokenized_titles, nlp=nlp)
                            lemmatized_titles = change_char(lemmatized_titles)
                            no_stops = remove_stops(lemmatized_titles, stops)
                            no_stops = [l for l in no_stops if len(l) != 0]
                            no_stops = rem_single_char(no_stops)

                            bigram, dictionary, corpus = prepare_input(no_stops)

                            max_topics = set_max_topics(no_stops)
                            min_topics = 2
                            step = 2

                            x = range(min_topics, max_topics, step)
                            coherence_values = compute_c_v(
                                dictionary=dictionary,
                                corpus=corpus,
                                texts=no_stops,
                                min_topics=min_topics,
                                max_topics=max_topics,
                                step=step,
                            )

                            best_result_index = coherence_values.index(
                                max(coherence_values)
                            )
                            ldamodel = LdaModel(
                                corpus=corpus,
                                num_topics=x[best_result_index],
                                id2word=dictionary,
                                update_every=1,
                                passes=10,
                                per_word_topics=True,
                            )

                            model = pyLDAvis.gensim_models.prepare(
                                ldamodel, corpus, dictionary
                            )
                            model_name = "G_" + file_name + ".html"
                            save_model(model, "../html/", model_name)

                            if "Contribute" in file:
                                data_generic = slice_data(no_stops)

                                if not os.path.exists(
                                    "/apps/trapholt/generic_models/GG_model.txt"
                                ):
                                    with open(
                                        "/apps/trapholt/generic_models/GG_model.txt",
                                        "w",
                                    ) as fp:
                                        pass

                                txt_append(
                                    "/apps/trapholt/generic_models/GG_model.txt",
                                    data_generic,
                                )
                    else:
                        old_name = "../json/" + file
                        new_name = "../json/" + file.replace("G", "FB")
                        os.rename(old_name, new_name)

                except (ValueError, KeyError, TypeError) as error:

                    if not os.path.exists("../not_processed/"):
                        os.mkdir("../not_processed/")
                    else:
                        pass

                    not_processed(
                        folder_path="../json/",
                        file_name=file,
                        folder_to_save="../not_processed/",
                    )

            else:
                continue

            if os.path.exists(file_path):
                os.remove(file_path)

    time.sleep(60)
