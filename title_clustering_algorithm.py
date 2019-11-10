import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

from constants import Constants
from corpus import Corpus
# from word2vec.cbow_embedding_model import CbowEmbeddingGenerator
import gensim, logging
import os
import pickle


class TitleClusteringAlgorithm:
    def __init__(self, corpus, embedding_generator):
        self.corpus = corpus
        self.emdeddingGenerator = embedding_generator

    def run(self, dictionary_size=300):
        # Part 1: Dictionary Learning
        # Get the embedding vectors; normalize them and project onto the unit sphere. Apply k-means there;
        # on the unit sphere Euclidian distance acts as a meaningful proxy to the angular distance:
        # c = sqrt(a^2 + b^2 - 2*a*b*cos(alpha))
        # c = sqrt(2 - 2*cos(alpha)) since we are on the unit sphere (a=1 and b=1)
        # c increases monotonically as alpha goes from 0 to pi.
        # We would like to use k-medoids instead of k-means with cosine distance, but k-medoids is very slow compared
        # to k-means. K-means operates with Euclidean distance, which is not suitable for comparing word embeddings due
        # to the sparsity in the vector space model. But k-means can be a reasonable choice after projecting the
        # word embeddings onto the unit sphere due to its similar behavior to the angular distance.
        embeddings = np.copy(self.emdeddingGenerator.model.wv.vectors)
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.reciprocal(embedding_norms) * embeddings
        kmeans = KMeans(n_clusters=dictionary_size, random_state=0)
        kmeans.fit(normalized_embeddings)
        dictionary = np.copy(kmeans.cluster_centers_)
        counter = Counter(kmeans.labels_)
        dictionary_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
        normalized_dictionary = np.reciprocal(dictionary_norms) * dictionary
        # Part 2: Assign words to the most similar clusters
        clusters = {i: [] for i in range(dictionary_size)}
        cluster_names = {}
        forward_index = {word: obj.index for word, obj in self.emdeddingGenerator.model.wv.vocab.items()}
        for word, idx in forward_index.items():
            word_vector = normalized_embeddings[idx, :]
            similarities = np.dot(normalized_dictionary, np.expand_dims(word_vector, axis=1))
            max_idx = np.argmax(similarities)
            clusters[max_idx].append(word)
        print("X")
        clusters_with_freqs = {}
        for cluster_id, cluster_words in clusters.items():
            words_and_freqs = [(word, self.corpus.vocabularyFreqs[word]) for word in cluster_words]
            sorted_words_and_freqs = sorted(words_and_freqs, key=lambda tpl: tpl[1], reverse=True)
            clusters_with_freqs[cluster_id] = sorted_words_and_freqs
        print("X")
