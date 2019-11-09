import numpy as np
import gensim, logging
import os


class EmbeddingGenerator:
    def __init__(self, corpus):
        self.corpus = corpus

    def train_model(self, min_freq, epoch_count):
        sentences = [sentence.tolist() for sentence in self.corpus.clearedTitles]
        model = gensim.models.Word2Vec(sentences, min_count=min_freq, iter=epoch_count)
        path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                            "word2vec_model_{0}_min_freq_{1}".format(epoch_count, min_freq)))
        model.save(path)
        self.corpus.validate(model)

    def load_model(self, min_freq, epoch_count):
        path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                            "word2vec_model_{0}_min_freq_{1}".format(epoch_count, min_freq)))
        model = gensim.models.Word2Vec.load(path)
        self.corpus.validate(model)
        return model
