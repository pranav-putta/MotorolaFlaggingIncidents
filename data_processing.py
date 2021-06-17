import re
from pprint import pprint

import pandas as pd
import pyLDAvis
import spacy
import nltk
import gensim.corpora as corpora

from nltk.corpus import stopwords
from nltk.corpus import names
import string

import gensim.models
from gensim import utils
import pyLDAvis.gensim_models

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
    lda_processed = lambda: pd.read_pickle('data/lda_processed.pkl')
    predictions = lambda: pd.read_pickle('output/predictions.pkl')

    my_stopwords = None

    @staticmethod
    def get_stopwords():
        if DataProcess.my_stopwords is None:
            males = [name.lower() for name in names.words('male.txt')]
            females = [name.lower() for name in names.words('female.txt')]
            DataProcess.my_stopwords = set(stopwords.words('english') + males + females)

        return DataProcess.my_stopwords

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
            sentences = [re.sub(r'\S*@\S*\s?', '', s) for s in sentences]  # remove emails
            sentences = [s.translate(str.maketrans('', '', string.digits)) for s in sentences]  # remove digits
            sentences = [s.lower().strip() for s in sentences]  # lowercase all words and strip any spaces

            # remove stop words
            sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            sentences = [' '.join([w for w in sentence if w not in DataProcess.get_stopwords()]) for sentence in
                         sentences]

            return ' '.join(sentences)

        except Exception:
            # happens if there are null entries in the dataset, just ignore for now
            # todo: clean dataset to remove these entries
            return ''


class W2VCorpus:
    """ Class that manages word similarity vectorization. Used for training the gensim model """
    file = 'output/gensim-model-0.8'

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
        corpus = W2VCorpus(data)
        model = gensim.models.Word2Vec(sentences=corpus, )
        model.save(W2VCorpus.file)

    @staticmethod
    def load_gensim_model():
        model = gensim.models.Word2Vec.load(W2VCorpus.file)
        return model


class LDACorpus:
    def __init__(self, data: pd.DataFrame):
        texts = data[key_short_descriptions]
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(nltk.word_tokenize(text))

        self.data = tokenized_texts
        self.lda_model = None
        self.corpus = None
        self.id2word = None

    def process(self):
        print('...processing data...')
        bigram = gensim.models.Phrases(self.data, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[self.data], threshold=100)

        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # bigram data
        data = [bigram_mod[doc] for doc in self.data]

        lemmatizer = nltk.WordNetLemmatizer()
        data = [[lemmatizer.lemmatize(word) for word in doc] for doc in data]
        # data = [[word for (word, pos) in nltk.pos_tag(doc) if pos[0] == 'N'] for doc in data]
        df = pd.DataFrame({'texts': data})
        df.to_pickle('data/lda_processed.pkl')
        print('...finished processing data...')

    def train(self):
        id2word = corpora.Dictionary(self.data)

        texts = self.data
        corpus = [id2word.doc2bow(text) for text in texts]
        print('... training LDA Model ...')
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=20,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        lda_model.save('output/lda_model')
        self.lda_model, self.corpus, self.id2word = lda_model, corpus, id2word
        print('... finished training LDA Model! ...')

    @staticmethod
    def visualize(model: gensim.models.LdaModel):
        data = list(DataProcess.lda_processed()['texts'])
        id2word = corpora.Dictionary(data)

        texts = data
        corpus = [id2word.doc2bow(text) for text in texts]
        vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary=id2word)
        pyLDAvis.save_html(vis, 'lda.html')

    @staticmethod
    def train_lda_model(process=False):
        lda = LDACorpus(DataProcess.processed())
        if process:
            lda.process()
        else:
            lda.data = list(DataProcess.lda_processed()['texts'])

        lda.train()

    @staticmethod
    def load_lda_model():
        return gensim.models.LdaModel.load('output/lda_model')


def construct_lda_word_model(process=False):
    if process:
        DataProcess.process(DataProcess.raw())
    LDACorpus.train_lda_model(process=process)


def construct_word_model(training_ratio=0.8):
    DataProcess.process(DataProcess.raw())
    data = DataProcess.processed()
    W2VCorpus.train_gensim_model(data[:int(len(data.index) * training_ratio)])
    print(f'..done constructing word embedding [training_ratio={training_ratio}]')
