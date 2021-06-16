import re

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

import gensim.models
from gensim import utils

key_short_descriptions = 'inc_short_description'


class DataProcess:
    """
    Any raw data preprocessing goes through here
    process() -- preprocess all entries in dataset and save to file
    process_text() -- preprocess single string
    """

    """pull raw data"""
    raw = lambda: pd.read_csv('data/raw.csv')
    """pull processed data"""
    processed = lambda: pd.read_pickle('data/processed.pkl')

    @staticmethod
    def process(data: pd.DataFrame):
        """
        processes textual data from the dataset
        :param data: data set to explore
        :return:
        """
        data[key_short_descriptions] = data[key_short_descriptions].apply(DataProcess._process_text)
        data.to_pickle('data/processed.pkl')
        return data

    @staticmethod
    def _process_text(text: str):
        try:
            sentences: list[str] = nltk.sent_tokenize(text)
            # lemmatizer = nltk.WordNetLemmatizer() # not using bc word2vec does this internally
            sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in
                         sentences]  # remove punctuation
            sentences = [re.sub(r'\w*\d\w*', '', s) for s in sentences]  # remove words with mix of numbers "coreIDs"
            sentences = [s.translate(str.maketrans('', '', string.digits)) for s in sentences]  # remove digits
            sentences = [s.lower().strip() for s in sentences]  # lowercase all words and strip any spaces

            # remove stop words
            stop_words = set(stopwords.words('english'))
            sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            sentences = [' '.join([w for w in sentence if w not in stop_words]) for sentence in sentences]

            return ' '.join(sentences)

        except Exception:
            # happens if there are null entries in the dataset, just ignore for now
            # todo: clean dataset to remove these entries
            return ''


class Corpus:
    """ Class that manages word similarity vectorization. Used for training the gensim model """
    file = 'output/gensim-model'

    def __init__(self, data: pd.DataFrame):
        # aggregate all sentences into one array
        texts = data[key_short_descriptions]
        self.sentences = texts.to_numpy()

    def __iter__(self):
        for sentence in self.sentences:
            if isinstance(sentence, str):
                yield utils.simple_preprocess(sentence)
            else:
                continue

    @staticmethod
    def train_gensim_model(data):
        corpus = Corpus(data)
        model = gensim.models.Word2Vec(sentences=corpus)
        model.save(Corpus.file)

    @staticmethod
    def load_gensim_model():
        model = gensim.models.Word2Vec.load(Corpus.file)
        return model


def construct_word_model():
    DataProcess.process(DataProcess.raw())
    Corpus.train_gensim_model(DataProcess.processed())
