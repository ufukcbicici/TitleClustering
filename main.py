import numpy as np

from constants import Constants
from corpus import Corpus
# from word2vec.cbow_embedding_model import CbowEmbeddingGenerator
import gensim, logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_gensim(corpus):
    sentences = [sentence.tolist() for sentence in corpus.clearedTitles]
    vocabulary = corpus.vocabulary
    model = gensim.models.Word2Vec(sentences, min_count=1, iter=Constants.EPOCH_COUNT)
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                        "word2vec_model_{0}".format(Constants.EPOCH_COUNT)))
    model.save(path)
    corpus.validate(model)


def main():
    # test_io()
    corpus = Corpus()
    # corpus.build_corpus()
    corpus.load_corpus()
    train_gensim(corpus)
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
