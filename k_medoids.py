import numpy as np


class KMedoids:
    def __init__(self, distance_func):
        self.distance = distance_func

    def assign_to_clusters(self, vocabulary, medoids):
        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        total_cost = 0
        for idx, word in enumerate(vocabulary):
            medoid_distances = [(medoid, self.distance(word, medoid)) for medoid in medoids]
            medoid_distances = sorted(medoid_distances, key = lambda tpl: tpl[1])
            closest_medoid = medoid_distances[0][0]
            clusters[closest_medoid].append(word)
            total_cost += medoid_distances[0][1]

        return clusters, total_cost

    def run(self, vocabulary, cluster_count, max_iter=100):
        iter_count = 0
        vocabulary = np.array(vocabulary)
        # Step 1: Build; randomly select "cluster_count" medoids
        medoids = np.random.choice(vocabulary, cluster_count, replace=False)
        clusters, total_cost = self.assign_to_clusters(vocabulary=vocabulary, medoids=medoids)
        print("X")

    # @staticmethod
    # def k_medoids(corpus, similiarities, cluster_count, max_iter=100):
    #     iter_count = 0
    #     vocabulary = np.array(corpus.vocabulary)
    #     # Step 1: Build; randomly select "cluster_count" medoids
    #     medoids = np.random.choice(vocabulary, cluster_count, replace=False)
    #
    #     while True:
