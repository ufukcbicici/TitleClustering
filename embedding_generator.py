import numpy as np
import gensim, logging
import os
import pickle


class EmbeddingGenerator:
    def __init__(self, corpus):
        self.corpus = corpus4










































        11111
        self.model = None
        self.minFreq = None
        self.epochCount = None
        self.similarityMatrix = None
        self.distanceMatrix = None

    def train_model(self, min_freq, epoch_count):
        self.minFreq = min_freq
        self.epochCount = epoch_count
        sentences = [sentence.tolist() for sentence in self.corpus.clearedTitles]
        self.model = gensim.models.Word2Vec(sentences, min_count=min_freq, iter=epoch_count)
        path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                            "word2vec_model_{0}_min_freq_{1}".format(epoch_count, min_freq)))
        self.model.save(path)
        self.corpus.validate(self.model)

    def build_similarity_matrix(self):
        assert (self.model is not None and self.epochCount is not None and self.minFreq is not None)
        unnormalized_matrix = np.dot(self.model.wv.vectors, self.model.wv.vectors.T)
        norms = np.linalg.norm(self.model.wv.vectors, axis=1, keepdims=True)
        norms_matrix = np.dot(norms, norms.T)
        self.similarityMatrix = unnormalized_matrix * np.reciprocal(norms_matrix)
        # Small unit test
        # sample_words = np.random.choice(list(self.model.wv.vocab.keys()), 100)
        # distances0 = []
        # distances1 = []
        # for word1 in sample_words:
        #     for word2 in sample_words:
        #         dist0 = self.model.similarity(word1, word2)
        #         dist1 = self.similarityMatrix[self.model.wv.vocab[word1].index, self.model.wv.vocab[word2].index]
        #         distances0.append(dist0)
        #         distances1.append(dist1)
        # assert np.allclose(np.array(distances0), np.array(distances1))
        pickle.dump(self.similarityMatrix,
                    open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "saved_data", "similarity_matrix_{0}_min_freq_{1}.sav"
                                                      .format(self.epochCount, self.minFreq))), 'wb'))

    def build_distance_matrix(self):
        self.distanceMatrix = 1.0 - (0.5 * (self.similarityMatrix + 1.0))

    def distance_func(self, word1, word2):
        idx1 = self.model.wv.vocab[word1].index
        idx2 = self.model.wv.vocab[word2].index
        distance = self.distanceMatrix[idx1, idx2]
        return distance

    def load_model(self, min_freq, epoch_count):
        self.minFreq = min_freq
        self.epochCount = epoch_count
        path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                            "word2vec_model_{0}_min_freq_{1}".format(epoch_count, min_freq)))
        self.model = gensim.models.Word2Vec.load(path)
        self.corpus.validate(self.model)

    # def load_similarity_matrix(self):
    #     assert (self.model is not None and self.epochCount is not None and self.minFreq is not None)
    #     self.similarityMatrix = pickle.load(
    #         open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                                           "saved_data", "similarity_matrix_{0}_min_freq_{1}.sav"
    #                                           .format(self.epochCount, self.minFreq))), 'rb'))