import pathlib
from pathlib import Path
from turtle import st
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
from typing import List


def remove_mentions(data: List[str]) -> List[str]:
    """
    Removes data instances that contain mentions from corpus.

    Args:
        data (List[str]): The dataset to clean from mentions.

    Returns:
        List[str]: The input data without mentions.
    """

    dataset = []

    for text in data:
        if text.startswith("@"):
            text = text.replace(text, "")
        dataset.append(text)

    return dataset


def change_letter(data: List[str]) -> List[str]:
    """
    Substitutes Danish special letters with relevant equivalents.

    Args:
        data (List[str]): The dataset to change letters in.

    Returns:
        List[str]: The input data with Danish special letters substituted.
    """

    dataset = [re.sub("ø", "oe", text) for text in data]
    dataset = [re.sub("æ", "ae", text) for text in dataset]
    dataset = [re.sub("å", "aa", text) for text in dataset]
    dataset = [re.sub("Ø", "oe", text) for text in dataset]
    dataset = [re.sub("Æ", "ae", text) for text in dataset]
    dataset = [re.sub("Å", "aa", text) for text in dataset]
    dataset = [re.sub("ü", "ue", text) for text in dataset]
    dataset = [re.sub("Ü", "ue", text) for text in dataset]
    dataset = [re.sub("ä", "ae", text) for text in dataset]
    dataset = [re.sub("Ä", "ae", text) for text in dataset]
    dataset = [re.sub("ö", "oe", text) for text in dataset]
    dataset = [re.sub("Ö", "oe", text) for text in dataset]

    return dataset


def clean_text(data: List[str]) -> List[str]:
    """
    Cleans text from punctuation, URLs, special characters and lowercases.

    Args:
        data (List[str]): The dataset to clean.

    Returns:
        List[str]: The cleaned input data.
    """

    no_urls = [re.sub(r"http\S+", "", text) for text in data]
    no_special_ch = [
        re.sub(r"(#[A-Za-z]+)|(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
        for text in no_urls
    ]
    no_special_ch = [text.replace("\n", " ") for text in no_special_ch]
    lowercased_str = [text.lower() for text in no_special_ch]
    cleaned_text = [re.sub(" +", " ", text) for text in lowercased_str]
    cleaned_text = [text.strip() for text in cleaned_text]

    return cleaned_text


def tokenize(data: List[str]) -> List[str]:
    """
    Tokenizes text data.

    Args:
        data (List[str]): The dataset to tokenize.

    Returns:
        List[str]: The list of tokens.
    """

    tokens = [word_tokenize(text) for text in data]

    return tokens


def lemmatize_str(text: List[str], nlp) -> List[str]:
    """
    Lemmatizes text.

    Args:
        text (List[str]): The dataset to lemmatize.
        nlp: spaCy nlp

    Returns:
        List[str]: The list with lemmas.
    """

    lemmas = [token.lemma_ for token in nlp(text)]

    return lemmas


def lemmatize(dataset: List[str], nlp) -> List[str]:

    lemmas = []

    for text in dataset:
        lemma = [lemmatize_str(token, nlp=nlp) for token in text]
        lemmas.append([item for sublist in lemma for item in sublist])

    return lemmas


def remove_stops(dataset: List[str], stopwords: List[str]) -> List[str]:
    """
    Removes stopwords from text.

    Args:
        dataset (List[str]): The data to remove stops from.
        stopwords (List[str]): The list of stopwords.

    Returns:
        List[str]: The input dataset without stopwords.
    """

    no_stopwords = []

    for text in dataset:
        tmp_list = []
        for token in text:
            if token not in stopwords:
                tmp_list.append(token)
        no_stopwords.append(tmp_list)

    return no_stopwords


def remove_empty_str(data: List[str]) -> List[str]:
    """
    Removes empty strings.

    Args:
        data (List[str]): The dataset to clean from empty str.

    Returns:
        List[str]: The input data without empty str.
    """
    result = [text for text in data if text != ""]
    return result


def change_char(dataset: List[str]) -> List[str]:

    for i in range(len(dataset)):
        dataset[i] = change_letter(dataset[i])
    return dataset


def rem_single_char(dataset: List[str]) -> List[str]:
    """
    Removes 1 letter instances from text corpus.

    Args:
        dataset (List[str]): The data to clean.

    Returns:
        List[str]: Cleaned input dataset.
    """

    for text in dataset:
        for token in text:
            if len(token) < 2:
                text.remove(token)
    return dataset
