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
        self.normalizedEmbeddings = None
        self.normalizedDictionary = None
        self.clusters = None
        self.clustersWithFreqs = None

    def run(self, dictionary_size=300, bow_method="frequencies"):
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
        self.normalizedEmbeddings = np.reciprocal(embedding_norms) * embeddings
        self.normalizedDictionary = self.get_cluster_centers(dictionary_size=dictionary_size)

        # Part 2: Assign words to the most similar clusters
        clusters = self.assign_to_clusters(dictionary_size=dictionary_size)

        # Part 3: From most frequent to least, pick %p percent of frequency. If this is more than n words, crop it.
        cluster_names, clusters_with_freqs = self.get_cluster_names(clusters=clusters)

        # Part 4: Build bag-of-words vector out of dictionaries

    def assign_titles_to_clusters(self, wv, method, cluster_names):
        vocab = wv.vocab
        title_assignments = {}
        for title in self.corpus.clearedTitles:
            # Bow methods:
            # "frequencies": The number of times a word w belonging to the i.th cluster is coded in the i. dimension.
            # "normalized_frequencies": Same as "frequencies" but the vector is normalized to make the sum 1.
            # "weighted_distances" For each for w in the title, cosine similarities with the i.th cluster center is measured and averaged.
            #  This is encoded into the i. dimension.
            # bow_vector = np.zeros((self.normalizedDictionary.shape[0], ))
            word_indices = [vocab[w].index for w in title if w in vocab]
            title_str = " ".join(title)
            if len(word_indices) == 0:
                continue
            title_arr = []
            for idx in word_indices:
                title_arr.append(self.normalizedEmbeddings[idx, :])
            title_arr = np.stack(title_arr, axis=0)
            distances_to_dictionary = np.dot(self.normalizedDictionary, title_arr.T)
            if method == "mean_of_similarities_to_centers":
                mean_similarities = np.mean(distances_to_dictionary, axis=1)
                cluster_id = np.argmax(mean_similarities)
            elif method == "max_of_individual_similarities":
                argmax_clusters = np.argmax(distances_to_dictionary, axis=0)
                coeffs = np.zeros_like(distances_to_dictionary)
                coeffs[np.array([i for i in range(distances_to_dictionary.shape[1])]), argmax_clusters] = 1.0
                selected_distances = coeffs * distances_to_dictionary
                total_distances = np.sum(selected_distances, axis=1)
                cluster_id = np.argmax(total_distances)
            else:
                raise NotImplementedError()
            cluster_name = cluster_names[cluster_id]
            title_assignments[title_str] = cluster_name
        return title_assignments

    def get_cluster_centers(self, dictionary_size):
        kmeans = KMeans(n_clusters=dictionary_size, random_state=0)
        kmeans.fit(self.normalizedEmbeddings)
        dictionary = np.copy(kmeans.cluster_centers_)
        counter = Counter(kmeans.labels_)
        dictionary_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
        normalized_dictionary = np.reciprocal(dictionary_norms) * dictionary
        return normalized_dictionary

    def assign_to_clusters(self, dictionary_size):
        clusters = {i: [] for i in range(dictionary_size)}
        forward_index = {word: obj.index for word, obj in self.emdeddingGenerator.model.wv.vocab.items()}
        for word, idx in forward_index.items():
            word_vector = self.normalizedEmbeddings[idx, :]
            similarities = np.dot(self.normalizedDictionary, np.expand_dims(word_vector, axis=1))
            max_idx = np.argmax(similarities)
            clusters[max_idx].append(word)
        return clusters

    def get_cluster_names(self, clusters, p=0.8, word_limit=10):
        clusters_with_freqs = {}
        cluster_names = {}
        for cluster_id, cluster_words in clusters.items():
            words_and_freqs = [(word, self.corpus.vocabularyFreqs[word]) for word in cluster_words]
            sorted_words_and_freqs = sorted(words_and_freqs, key=lambda tpl: tpl[1], reverse=True)
            clusters_with_freqs[cluster_id] = sorted_words_and_freqs
            freqs = np.array([tpl[1] for tpl in sorted_words_and_freqs])
            cum_freqs = np.cumsum(freqs)
            cum_normalized_freqs = cum_freqs / np.sum(freqs)
            selected_word_ids = cum_normalized_freqs[cum_normalized_freqs <= p][0: word_limit]
            cluster_name = ",".join([sorted_words_and_freqs[i][0] for i in range(selected_word_ids.shape[0])])
            cluster_names[cluster_id] = cluster_name
        return cluster_names, clusters_with_freqs

    def load_clustering_data(self):
        self.normalizedDictionary = pickle.load(
            open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                              "normalizedDictionary.sav")), 'rb'))
        self.clusters = pickle.load(
            open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                              "clusters.sav")), 'rb'))
        self.clustersWithFreqs = pickle.load(
            open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data",
                                              "clustersWithFreqs.sav")), 'rb'))


