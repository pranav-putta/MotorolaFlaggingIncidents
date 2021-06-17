from pprint import pprint

import keras
import pandas as pd

import data_processing as dp
import train as tr
import numpy as np
import nltk

if __name__ == '__main__':
    # View processed and raw data here
    data: pd.DataFrame = dp.DataProcess.processed()
    data = data[data["sus_u_employee_type"] == "R"]
    # raw = dp.DataProcess.raw()
    # lda = dp.DataProcess.lda_processed()
    # predictions: pd.DataFrame = dp.DataProcess.predictions()
    # predictions.to_csv('predictions1.csv')

    # 1. Construct the word model
    # dp.construct_word_model()
    # dp.construct_lda_word_model(process=True)
    model = dp.W2VCorpus.load_gensim_model()
    # dp.LDACorpus.visualize(model)
    # tr.train_rnn(model, data)
    tr.validate_rnn(model, data, train_ratio=0.4)
    df = tr.predict_neural_network(model, data.index[int(0.4 * len(data.index)):])
    # df = dp.DataProcess.predictions()

    # 2. Construct clusterings using word model.
    # tr.cluster_gensim_model(model)
    # tr._1cluster_gensim_model(model)

    # 3. Train neural network
    # tr.train_neural_network(model)
