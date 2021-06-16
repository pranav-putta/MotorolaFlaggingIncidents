import pandas as pd
import sklearn.metrics
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.neighbors import NearestNeighbors
import data_processing as dp
import json

import numpy as np
from pprint import pprint
from nltk import word_tokenize

from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense


def cos_dist(x, y):
    """ constructs cosine distance of two vectors
        Formula as follows:
        cos(theta) = X^T * Y / |X| * |Y|
        distance = 1 - cos^3(theta) ## custom mapping function,
                                     can be modified as long as
                                     lim(x->1) = 0 and lim(x->-1) >= 0
    """
    similarity = (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))  # cosine theta
    transform = 1 - similarity ** 3  # mapping function f(x) = 1 - x^3
    return transform


def display_labels(model: Word2Vec):
    """
    pretty print clusters for word model
    :param model:
    :return:
    """
    labels = np.fromfile('cluster_labels.csv', sep=',')
    words = np.array(model.wv.index_to_key)
    counts = {}
    for i in range(len(labels)):
        cluster = labels[i]
        if cluster in counts.keys():
            counts[cluster].append(words[i])
        else:
            counts[cluster] = [words[i]]
    pprint(counts)
    # print(json.dumps(counts))


def cluster_gensim_model(model: Word2Vec):
    """
    Affinity Propagation clustering algorithm
    :param model: word model (gensim)
    :return:
    """
    model = model.wv
    words = model.index_to_key
    vectors = np.array(model.vectors)
    dist_matrix = np.array([[cos_dist(v_i, v_j) for v_i in vectors] for v_j in vectors])

    clustering = AffinityPropagation(verbose=True, damping=0.7).fit(dist_matrix)
    labels = np.array(clustering.labels_)

    labels.tofile('cluster_labels.csv', ',')

    counts = {}
    for i in range(len(labels)):
        cluster = labels[i]
        if cluster in counts.keys():
            counts[cluster].append(words[i])
        else:
            counts[cluster] = [words[i]]
    pprint(counts)


def train_neural_network(model: Word2Vec):
    """
    Classification neural network, currently 3 layer 1000-500 softmax activation network
    * text is encoded as a count of clusters
        * for example processed text might be "help google password". The associated clusters might be
         "google" and "accounts"
        * this is represented as [ 0 0 0 1 ... 0 0 1 ... 0 0].
          Each index stands for a cluster, this is one hot encoding with counting
    * output is a list of probabilities for each assignment group
    :param model:
    :return:
    """
    labels = np.fromfile('cluster_labels.csv', sep=',')
    num_clusters = len(np.unique(labels))
    words = np.array(model.wv.index_to_key)
    cluster_dict = {words[i]: int(labels[i]) for i in range(len(words))}
    data: pd.DataFrame = dp.DataProcess.processed()

    data_labels = data['inc_assignment_group']
    groups = data_labels.unique()
    ag = {groups[i]: i for i in range(len(groups))}
    Y = []
    for data_label in data_labels:
        label = np.zeros(len(groups))
        try:
            label[ag[data_label]] = 1
        except Exception:
            pass
        Y.append(label)
    Y = np.array(Y)

    raw = data['inc_short_description'].to_numpy()
    # vectorize input
    X = []
    for description in raw:
        desc_words = word_tokenize(description)
        clusters = np.zeros(num_clusters + 1)
        for word in desc_words:
            if word in cluster_dict.keys():
                cluster = cluster_dict[word]
                clusters[cluster] += 1
            else:
                clusters[num_clusters] += 1
        X.append(clusters)
    X = np.array(X)

    model = Sequential()
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(len(groups), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, Y, epochs=100, batch_size=120)
    model.save('dnn.model')


def predict_neural_network(model: Word2Vec):
    """
    Placeholder prediction function
    :param model:
    :return:
    """
    labels = np.fromfile('cluster_labels.csv', sep=',')
    num_clusters = len(np.unique(labels))
    words = np.array(model.wv.index_to_key)
    cluster_dict = {words[i]: int(labels[i]) for i in range(len(words))}
    data: pd.DataFrame = dp.DataProcess.processed()
    model: keras.Sequential = keras.models.load_model('dnn.model')

    data_labels = data['inc_assignment_group']
    groups = data_labels.unique()

    raw = data['inc_short_description'].to_numpy()
    # vectorize input
    X = []
    for description in raw:
        desc_words = word_tokenize(description)
        clusters = np.zeros(num_clusters + 1)
        for word in desc_words:
            if word in cluster_dict.keys():
                cluster = cluster_dict[word]
                clusters[cluster] += 1
            else:
                clusters[num_clusters] += 1
        X.append(clusters)
    X = np.array(X)
    sc = StandardScaler()
    X = sc.fit_transform(X)

    data_labels = data['inc_assignment_group']

    ct = 100
    result = (model.predict(X[:ct + 1]))
    s = np.argsort(result)
    for x in range(ct):
        print(raw[x], data_labels[x], [groups[i] for i in s[x][:10]])


# 1st attempt
def _1cluster_gensim_model(model: Word2Vec):
    """DBSCAN algorithm attempt -- did not work well, creating really big clusters no correlation"""
    model = model.wv
    words = model.index_to_key
    vectors = np.array(model.vectors)
    indices = np.random.choice(np.arange(0, len(vectors)), 1000, False)
    vectors = vectors[indices]
    # words = ['password', 'pw', 'pwd', 'aws', 'apple', 'llc']
    # vectors = [model[word] for word in words]
    db = DBSCAN(metric=cos_dist, eps=0.5, min_samples=3, algorithm='ball_tree').fit(vectors)
    labels = db.labels_
    print(np.unique(db.labels_))

    counts = {}
    for i in range(len(labels)):
        cluster = labels[i]
        if cluster in counts.keys():
            counts[cluster].append(words[i])
        else:
            counts[cluster] = [words[i]]
    pprint(counts)

    cores = [words[i] for i in db.core_sample_indices_]
    print(cores)

