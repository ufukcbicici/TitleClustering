import numpy as np

from constants import Constants
from corpus import Corpus
# from word2vec.cbow_embedding_model import CbowEmbeddingGenerator
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def test_io():
    corpus1 = Corpus()
    corpus1.clear_titles()
    corpus1.save_cleared_titles()

    corpus2 = Corpus()
    corpus2.load_cleared_titles()

    assert corpus1.clearedTitles.shape == corpus2.clearedTitles.shape
    for idx in range(corpus1.clearedTitles.shape[0]):
        assert np.array_equal(corpus1.clearedTitles[idx], corpus2.clearedTitles[idx])


def train_gensim(corpus):
    sentences = [title.tolist() for title in corpus.clearedTitles if len(title) >= 1]
    model = gensim.models.Word2Vec(sentences, min_count=1, iter=1000)
    print("X")


def main():
    # test_io()
    corpus = Corpus()
    corpus.clear_titles()
    corpus.build_corpus()
    corpus.read_cbow_data()
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