import sys
import pathlib
from pathlib import Path
import os
import datetime as dt
import time
import json
import spacy
import codecs
from typing import List
import gensim

import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation


def load_json(path: str) -> List[str]:
    """
    Loads json file and saves in a list.

    Args:
       path (str): Path to a file.

    Returns:
        dataset (List[str]): The data saved in a list.
    """

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def save_txt(dataset: List[str], file_path: str):
    """
    Writes and saves data to txt file.

    Args:
        dataset (List[str]): The data to save.
        file_path (str): Path to save txt file.
    """

    f = open(file_path, "w")

    for text in dataset:
        without_line_breaks = text.replace("\n", " ")
        without_line_breaks = without_line_breaks.replace("\r", " ")
        lines = without_line_breaks + "\n"
        f.write(lines)
    f.close()


def decode(path: str) -> List[str]:
    """
    Decodes data.

    Args:
        path (str): Path to a file with data.

    Returns:
        List[str]: Data saves into a list.
    """

    f = codecs.open(path, errors="ignore", encoding="utf-8")
    data = f.read()
    data = data.encode("latin1").decode("utf8")
    data = data.splitlines()

    return data


def save_model(model, folder_name: str, file_name: str):
    """
    Creates a folder and saves topic model in it.

    Args:
       model: pyLDAvis gensim model

       folder_name (str): The name of the folder to save a model in.

       file_name (str): The name of the html file with the model.
    """

    current_dir = pathlib.Path().resolve()
    path = os.path.join(current_dir, folder_name)

    if not os.path.exists(folder_name):
        os.mkdir(path)

    file_path = os.path.join(path, file_name)

    pyLDAvis.save_html(model, file_path)


def not_processed(file_name: str, folder_path: str, folder_to_save: str):
    """
    Changes the filename and save into a folder.

    Args:
        file_name (str): The name of the file to process and save.
        folder_path (str): The folder where the file is located.
        folder_to_save (str): The name of the folder to save the resulting file in.
    """

    old_name = folder_path + file_name
    file_name = file_name.split("_")[-1]
    new_name = folder_to_save + "Decoding_JSON_has_failed_" + file_name
    os.rename(old_name, new_name)
