import logging

from constants import Constants
from corpus import Corpus
from embedding_generator import EmbeddingGenerator
from title_clustering_algorithm import TitleClusteringAlgorithm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    # test_io()
    corpus = Corpus()
    # corpus.build_corpus()
    corpus.load_corpus()
    embedding_model = EmbeddingGenerator(corpus=corpus)
    # embedding_model.train_model(min_freq=Constants.MIN_FREQ, epoch_count=Constants.EPOCH_COUNT)
    embedding_model.load_model(min_freq=Constants.MIN_FREQ, epoch_count=Constants.EPOCH_COUNT)

    title_clustering_algorithm = TitleClusteringAlgorithm(corpus=corpus, embedding_generator=embedding_model)
    title_clustering_algorithm.run(dictionary_size=40)


main()
