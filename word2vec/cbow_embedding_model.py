import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.contrib.framework.python.framework import checkpoint_utils

from constants import Constants


class CbowEmbeddingGenerator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.trainContext = tf.placeholder(tf.int32, shape=[Constants.EMBEDDING_BATCH_SIZE,
                                                            2 * Constants.CBOW_WINDOW_SIZE])
        self.trainLabels = tf.placeholder(tf.int32, shape=[Constants.EMBEDDING_BATCH_SIZE, 1])
        self.contextWeights = tf.placeholder(tf.float32, shape=[Constants.EMBEDDING_BATCH_SIZE,
                                                                2 * Constants.CBOW_WINDOW_SIZE])
        self.globalStep = tf.Variable(0, trainable=False)
        self.rawEmbeddings = []
        self.weightedEmbeddings = []
        self.stackedEmbeddings = None
        self.averagedEmbeddings = None
        self.loss = None
        self.optimizer = None
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        vocabulary_size = self.corpus.get_vocabulary_size()
        embedding_size = Constants.EMBEDDING_SIZE
        window_size = Constants.CBOW_WINDOW_SIZE
        # These are the embeddings we are going to learn.
        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, Constants.EMBEDDING_SIZE], -1.0, 1.0),
                                      name="embeddings")
        # Softmax weights
        self.softmaxWeights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)),
                                          name="softmaxWeights")
        self.softmaxBiases = tf.Variable(tf.zeros([vocabulary_size]), name="softmaxBiases")

    def get_embeddings(self, sess):
        return self.embeddings.eval(session=sess)

    def build_network(self):
        vocabulary_size = self.corpus.get_vocabulary_size()
        window_size = Constants.CBOW_WINDOW_SIZE
        num_negative_sampling = Constants.NUM_NEGATIVE_SAMPLES
        # Build the operations
        for i in range(2 * window_size):
            embedding_vectors = tf.nn.embedding_lookup(self.embeddings, self.trainContext[:, i])
            self.rawEmbeddings.append(embedding_vectors)
            weighted_embedding_vectors = tf.expand_dims(self.contextWeights[:, i], axis=1) * embedding_vectors
            self.weightedEmbeddings.append(weighted_embedding_vectors)
        self.stackedEmbeddings = tf.stack(axis=0, values=self.weightedEmbeddings)
        self.averagedEmbeddings = tf.reduce_sum(self.stackedEmbeddings, axis=0, keep_dims=False)
        self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=self.softmaxWeights, biases=self.softmaxBiases,
                                           inputs=self.averagedEmbeddings, labels=self.trainLabels,
                                           num_sampled=num_negative_sampling, num_classes=vocabulary_size))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        iteration_count = 0
        epoch_count = Constants.EPOCH_COUNT
        batch_size = Constants.EMBEDDING_BATCH_SIZE
        losses = []
        sess.run(tf.global_variables_initializer())
        # self.contextGenerator.validate(corpus=self.corpus, embeddings=self.get_embeddings(sess=sess))
        for epoch_id in range(epoch_count):
            print("*************Epoch {0}*************".format(epoch_id))
            while True:
                context, targets, weights = self.corpus.get_next_batch(batch_size=batch_size)
                targets = np.reshape(targets, newshape=(targets.shape[0], 1))
                feed_dict = {self.trainContext: context, self.trainLabels: targets, self.contextWeights: weights}
                # results = sess.run([self.rawEmbeddings, self.weightedEmbeddings, self.averagedEmbeddings],
                #                    feed_dict=feed_dict)
                results = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                losses.append(results[1])
                if iteration_count % 1000 == 0:
                    print("Iteration:{0}".format(iteration_count))
                    mean_loss = np.mean(np.array(losses))
                    print("Loss:{0}".format(mean_loss))
                    losses = []
                iteration_count += 1
                if self.corpus.isNewEpoch:
                    # Save embeddings to HD
                    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "embeddings", "embedding_epoch{0}.ckpt".format(epoch_id)))
                    saver.save(sess, path)
                    # embeddings_arr = self.embeddings.eval(session=sess)
                    # self.corpus.validate(corpus=self.corpus, embeddings=self.get_embeddings(sess=sess))
                    break
        print("X")

    def test_embedding_network(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        batch_size = Constants.EMBEDDING_BATCH_SIZE
        context, targets, weights = self.corpus.get_next_batch(batch_size=batch_size)
        targets = np.reshape(targets, newshape=(targets.shape[0], 1))
        feed_dict = {self.trainContext: context, self.trainLabels: targets, self.contextWeights: weights}
        results = sess.run([self.rawEmbeddings, self.weightedEmbeddings, self.averagedEmbeddings],
                           feed_dict=feed_dict)
        embeddings = self.embeddings.eval(sess)
        for i in range(batch_size):
            w = weights[i, :]
            embeddings_i = embeddings[context[i, :]]
            weighted_embeddings_i = np.expand_dims(w, axis=1) * embeddings_i
            avg_embedding_manual = np.sum(weighted_embeddings_i, axis=0)
            avg_embedding_tf = results[2][i, :]
            assert np.allclose(avg_embedding_manual, avg_embedding_tf)

    # def load_embeddings(self):
    #     # pretrained_var_list = checkpoint_utils.list_variables("embeddings_epoch99.ckpt")
    #     source_array = checkpoint_utils.load_variable(checkpoint_dir=Constants.EMBEDDING_CHECKPOINT_PATH,
    #                                                   name="embeddings")
    #     tf.assign(self.embeddings, source_array).eval(session=self.sess)
