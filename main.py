import numpy as np

from constants import Constants
from corpus import Corpus
# from word2vec.cbow_embedding_model import CbowEmbeddingGenerator
import gensim, logging
import os
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_gensim(corpus):
    sentences = [sentence.tolist() for sentence in corpus.clearedTitles]
    vocabulary = corpus.vocabulary
    model = gensim.models.Word2Vec(sentences, min_count=1, iter=Constants.EPOCH_COUNT)
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                        "word2vec_model_{0}".format(Constants.EPOCH_COUNT)))
    model.save(path)
    corpus.validate(model)


def calculate_similarity_info(corpus, embedding_model):
    similarity_dict = {}
    for word_1 in corpus.vocabulary:
        for word_2 in corpus.vocabulary:
            similarity_dict[(word_1, word_2)] = 0.5 * (embedding_model.similarity(word_1, word_2) + 1.0)
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                        "similarity_dict.sav"))
    pickle.dump(similarity_dict, path)


def load_word2vec_model(corpus):
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                        "word2vec_model_{0}".format(Constants.EPOCH_COUNT)))
    model = gensim.models.Word2Vec.load(path)
    corpus.validate(model)


def k_medoids(corpus, similiarities, cluster_count, max_iter=100):
    iter_count = 0
    vocabulary = np.array(corpus.vocabulary)
    # Step 1: Build; randomly select "cluster_count" medoids
    medoids = np.random.choice(vocabulary, cluster_count, replace=False)
    clusters = {}
    while True:



    # Gr
    # for iteration in range(max_iter):




def main():
    # test_io()
    corpus = Corpus()
    # corpus.build_corpus()
    corpus.load_corpus()
    load_word2vec_model(corpus)
    # train_gensim(corpus)
    # sentences = [['first', 'sentence'], ['second', 'sentence']]
    # # train word2vec on the two sentences
    # model = gensim.models.Word2Vec(sentences, min_count=1)
    # # corpus.build_contexts()
    # # embedding_model = CbowEmbeddingGenerator(corpus=corpus)
    # # embedding_model.build_network()
    # # embedding_model.test_embedding_network()
    # # embedding_model.train()
    print("X")


main()
