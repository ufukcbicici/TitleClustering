import numpy as np
from corpus import Corpus
from word2vec.cbow_embedding_model import CbowEmbeddingGenerator


def test_io():
    corpus1 = Corpus()
    corpus1.clear_titles()
    corpus1.save_cleared_titles()

    corpus2 = Corpus()
    corpus2.load_cleared_titles()

    assert corpus1.clearedTitles.shape == corpus2.clearedTitles.shape
    for idx in range(corpus1.clearedTitles.shape[0]):
        assert np.array_equal(corpus1.clearedTitles[idx], corpus2.clearedTitles[idx])


def main():
    # test_io()
    corpus = Corpus()
    corpus.clear_titles()
    corpus.build_corpus()
    embedding_model = CbowEmbeddingGenerator(corpus=corpus)
    embedding_model.build_network()
    print("X")


main()