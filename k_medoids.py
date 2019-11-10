import numpy as np
import time


class KMedoids:
    def __init__(self, vocabulary, distance_matrix, distance_func):
        self.distanceMatrix = distance_matrix
        self.forwardIndex = {word: obj.index for word, obj in vocabulary.items()}
        self.backwardIndex = {obj.index: word for word, obj in vocabulary.items()}
        self.vocabulary = [0] * len(self.backwardIndex)
        for k, v in self.backwardIndex.items():
            self.vocabulary[k] = v
        self.vocabulary = np.array(self.vocabulary)
        self.distance = distance_func

    def assign_to_clusters(self, medoids):
        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        total_cost = 0
        # Get all indices for the medoids
        medoid_indices = np.array(sorted([self.forwardIndex[medoid] for medoid in medoids]))
        # Get all relevant distance columns for medoids
        distances_to_medoids = self.distanceMatrix[:, medoid_indices]
        # Calculate arg min columns
        arg_min_columns = np.argmin(distances_to_medoids, axis=1)
        # Get relevant medoid indices
        medoid_idx_per_word = medoid_indices[arg_min_columns]
        medoids_per_word = np.array([self.backwardIndex[medoid_idx] for medoid_idx in medoid_idx_per_word])
        # Assign each word to correspondig medoid's cluster
        for word_idx, medoid in enumerate(medoids_per_word):
            clusters[medoid].append(self.backwardIndex[word_idx])
        # Get total configuration cost
        min_distances = np.min(distances_to_medoids, axis=1)
        total_cost = np.sum(min_distances)
        return clusters, total_cost

    def assign_to_clusters_ex(self, list_of_medoids):
        # Get all indices for the medoids
        t0 = time.time()
        medoid_indices = np.stack([np.array(sorted([self.forwardIndex[medoid] for medoid in medoids]))
                                   for medoids in list_of_medoids], axis=0)
        distance_arrs = [self.distanceMatrix[:, medoid_indices[i, :]] for i in range(medoid_indices.shape[0])]
        t1 = time.time()
        distance_tensor = np.stack(distance_arrs, axis=0)
        # Calculate arg min columns
        arg_min_columns = np.argmin(distance_tensor, axis=2)
        ix = np.zeros_like(arg_min_columns)
        for i in range(medoid_indices.shape[0]):
            ix[i, :] = i
        min_distances = np.min(distance_tensor, axis=2)
        all_costs = np.sum(min_distances, axis=1)
        selected_medoid_indices_arr = medoid_indices[ix, arg_min_columns]
        selected_medoids_arr = self.vocabulary[selected_medoid_indices_arr]
        t2 = time.time()
        list_of_congifurations = []
        for i in range(len(list_of_medoids)):
            clusters = {medoid: [] for medoid in list_of_medoids[i]}
            for word_idx, medoid in enumerate(selected_medoids_arr[i, :]):
                clusters[medoid].append(self.backwardIndex[word_idx])
            list_of_congifurations.append((clusters, all_costs[i]))
        t3 = time.time()
        return list_of_congifurations

    def assign_to_clusters_old(self, medoids):
        vocabulary = list(self.forwardIndex.keys())
        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        total_cost = 0
        for idx, word in enumerate(vocabulary):
            medoid_distances = [(medoid, self.distance(word, medoid)) for medoid in medoids]
            medoid_distances = sorted(medoid_distances, key=lambda tpl: tpl[1])
            closest_medoid = medoid_distances[0][0]
            clusters[closest_medoid].append(word)
            total_cost += medoid_distances[0][1]

        return clusters, total_cost

    def run(self, cluster_count, max_iter=100):
        iter_count = 0
        # Step 1: Build; greedily select out of 100 random start configurations
        configurations = []
        for idx in range(100):
            medoids = np.random.choice(self.vocabulary, cluster_count, replace=False)
            config, cost = self.assign_to_clusters(medoids=medoids)
            configurations.append((config, cost))
        order_init_configurations = sorted(configurations, key=lambda tpl: tpl[1])
        curr_clusters, curr_cost = order_init_configurations[0]
        while True:
            print("Iteration:{0}".format(iter_count))
            print("Curr Cost:{0}".format(curr_cost))
            # For every non medoid word w and for every medoid wor m, swap w and m and measure
            # the cost of the configuration
            curr_medoids = set(curr_clusters.keys())
            best_configuration = None
            best_cost = np.infty
            for idx, w in enumerate(self.vocabulary):
                if (idx + 1) % 1 == 0:
                    print("Processed {0} words".format(idx + 1))
                if w in curr_medoids:
                    continue
                # config_arr_1 = []
                # list_of_medoids = []
                # t0 = time.time()
                # for m in curr_medoids:
                #     new_medoids = [_m for _m in curr_medoids if _m != m]
                #     new_medoids.append(w)
                #     new_config, new_cost = self.assign_to_clusters(new_medoids)
                    # list_of_medoids.append(new_medoids)
                    # config_arr_1.append((new_config, new_cost))
                # t1 = time.time()
                # config_arr_2 = self.assign_to_clusters_ex(list_of_medoids=list_of_medoids)
                # t2 = time.time()
                # assert len(config_arr_1) == len(config_arr_2)
                # for i in range(len(config_arr_1)):
                #     for k in config_arr_1[i][0].keys():
                #         set_1 = set(config_arr_1[i][0][k])
                #         set_2 = set(config_arr_2[i][0][k])
                #         assert set_1 == set_2
                # dist_arr1 = np.array([tpl[1] for tpl in config_arr_1])
                # dist_arr2 = np.array([tpl[1] for tpl in config_arr_2])
                # assert np.allclose(dist_arr1, dist_arr2)
                for m in curr_medoids:
                    new_medoids = [_m for _m in curr_medoids if _m != m]
                    new_medoids.append(w)
                    new_config, new_cost = self.assign_to_clusters(new_medoids)
                    if new_cost < best_cost:
                        best_configuration = new_config
                        best_cost = new_cost
            if best_cost < curr_cost:
                curr_clusters = best_configuration
                curr_cost = best_cost
                print("New Best Cost:{0}".format(curr_cost))
            else:
                break
            iter_count += 1
            if iter_count >= max_iter:
                break
        return curr_clusters, curr_cost

    # @staticmethod
    # def k_medoids(corpus, similiarities, cluster_count, max_iter=100):
    #     iter_count = 0
    #     vocabulary = np.array(corpus.vocabulary)
    #     # Step 1: Build; randomly select "cluster_count" medoids
    #     medoids = np.random.choice(vocabulary, cluster_count, replace=False)
    #
    #     while True:
