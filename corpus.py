import numpy as np
import pandas as pd
import regex
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from constants import Constants
from dblogger import DbLogger


class Corpus:
    FILTER_REGEX = "[^a-zA-Z\d\s\-,，\\/]|[\r\n]"
    WORD_SPLIT_REGEX = "[/\\-,，\s]"
    EMAIL_MATCHER = "[a-zA-Z0-9\_\.\+\-]++@[a-zA-Z0-9\-]++\.[a-zA-Z0-9\-\.]++"
    URL_MATCHER = "((?<=\s)pic\.|www\.|https?\:\s?)[^\s]++|[0-9a-z\.]+\.([Cc](?:[Oo](?:[Mm]))|N(?:ET|et)|TR|tr|e(?:du|u)|E(?:DU|U)|net|org|GOV|gov|ORG|fr|FR|az|DE|RU|de|ru)[^\'\"\s]*+"
    DATE_MATCHER = "(?<=[\s\n])[0-9]{2}+([\.\-\/])[0-9]{2}+\1[0-9]{4}+(?!\1)"
    NUMBER_MATCHER = "(?:(?<=\s)\-)?(?<![0-9\.]|(?<!\sNO)(?<!\sK)\:)[0-9]++(([\,][0-9]++)++|(\.[0-9]++)++(\,[0-9]++)?|(?=[^\.\,\:])|[\.\,]++(?=$|\s))"

    EMAIL_TOKEN = "EMAIL"
    URL_TOKEN = "URL"
    DATE_TOKEN = "DATE"
    NUMBER_TOKEN = "NUMBER"

    def __init__(self):
        self.clearedTitles = None
        self.vocabulary = None
        self.vocabularyFreqs = None
        self.labelEncoder = None
        self.embeddingContextsAndTargets = None
        self.currentIndex = None
        self.currentIndices = None
        self.isNewEpoch = True

    def save_cleared_titles(self):
        with open("cleared_titles.txt", "w") as file:
            for title in self.clearedTitles:
                title_unified = ""
                for word in title:
                    title_unified += word
                    title_unified += " "
                title_unified = title_unified[0:-1]
                file.write("{0}\n".format(title_unified))

    def load_cleared_titles(self):
        self.clearedTitles = []
        with open("cleared_titles.txt", "r") as file:
            title = file.readline()
            while title:
                words = regex.split("\s", title[:-1])
                if len(words) == 1 and words[0] == '':
                    self.clearedTitles.append(np.array([]))
                else:
                    self.clearedTitles.append(np.array(words))
                title = file.readline()
        self.clearedTitles = np.array(self.clearedTitles)

    def clear_titles(self):
        titles_df = pd.read_csv('titles.csv')
        # Clear data
        self.clearedTitles = []
        # First clear, then split
        # for i in range(titles_df.shape[0]):
        #     raw_title = titles_df.loc[i][0]
        #     cleared_title = re.sub(Corpus.FILTER_REGEX, "", raw_title)
        #     words = re.split("[\\-,，\s]", cleared_title)
        #     words = [word.capitalize() for word in words if word != ""]
        #     self.clearedTitles.append(np.array(words))
        #     print("Title {0}: {1}".format(i, self.clearedTitles[-1]))

        # First split then clear
        for i in range(titles_df.shape[0]):
            title = titles_df.loc[i][0]
            # Step 1: Lower Case
            title = title.lower()
            # Step 2: Find and Replace Emails
            title = regex.sub(Corpus.EMAIL_MATCHER, Corpus.EMAIL_TOKEN, title)
            # Step 3: Find and Replace URLs
            title = regex.sub(Corpus.URL_MATCHER, Corpus.URL_TOKEN, title)
            # Step 4: Find and Replace Dates
            title = regex.sub(Corpus.DATE_MATCHER, Corpus.DATE_TOKEN, title)
            # Step 5: Clean non-alphanumeric etc. characters
            title = regex.sub(Corpus.FILTER_REGEX, "", title)
            # Step 6: Split into words
            words = regex.split(Corpus.WORD_SPLIT_REGEX, title)
            # Step 7: Skip empty tokens
            words = [word for word in words if word != ""]
            # Step 8: Add to cleared titles
            self.clearedTitles.append(np.array(words))
            print("Title {0}: {1}".format(i, self.clearedTitles[-1]))
        self.clearedTitles = np.array(self.clearedTitles)

    def build_corpus(self):
        word_list = []
        for words in self.clearedTitles:
            if len(words) == 0:
                continue
            for word in words:
                word_list.append(word)
        self.vocabularyFreqs = Counter(word_list)
        self.vocabulary = list(self.vocabularyFreqs.keys())
        self.vocabulary.append("UNK")
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.vocabulary)

    def encode_context_data(self, context_arr_2D):
        context_arr_flat = np.reshape(context_arr_2D, newshape=(context_arr_2D.shape[0] * context_arr_2D.shape[1], ))
        label_ids = self.labelEncoder.transform(context_arr_flat)
        self.embeddingContextsAndTargets = np.reshape(label_ids, newshape=context_arr_2D.shape)

    def build_contexts(self):
        rows = []
        table_name = "cbow_context_window_{0}_table".format(Constants.CBOW_WINDOW_SIZE)
        # Delete cbow table
        DbLogger.delete_table(table=table_name)
        sequence_count = 0
        context_arr = []
        for title in self.clearedTitles:
            if len(title) <= 1:
                continue
            for token_index, target_token in enumerate(title):
                context = []
                # Harvest a context
                for delta in range(-Constants.CBOW_WINDOW_SIZE, Constants.CBOW_WINDOW_SIZE + 1):
                    t = delta + token_index
                    if t < 0 or t >= len(title):
                        context.append("UNK")
                    elif t == token_index:
                        assert target_token == title[token_index]
                        continue
                    else:
                        token = title[t]
                        context.append(token)
                context.append(target_token)
                assert len(context) == 2 * Constants.CBOW_WINDOW_SIZE + 1
                rows.append(tuple(context))
                context_arr.append(np.expand_dims(context, axis=0))
            sequence_count += 1
            if sequence_count % 1000 == 0:
                print("{0} sequences have been processed.".format(sequence_count))
            if len(rows) >= 100000:
                print("CBOW tokens written to DB.")
                DbLogger.write_into_table(rows=rows, table=table_name,
                                          col_count=2 * Constants.CBOW_WINDOW_SIZE + 1)
                rows = []
        if len(rows) > 0:
            DbLogger.write_into_table(rows=rows, table=table_name,
                                      col_count=2 * Constants.CBOW_WINDOW_SIZE + 1)
        context_arr_2D = np.concatenate(context_arr, axis=0)
        self.encode_context_data(context_arr_2D=context_arr_2D)

    def read_cbow_data(self):
        table_name = "cbow_context_window_{0}_table".format(Constants.CBOW_WINDOW_SIZE)
        condition = ""
        # for i in range(2 * Constants.CBOW_WINDOW_SIZE):
        #     condition += "Token{0} != -1".format(i)
        #     if i < 2 * GlobalConstants.CBOW_WINDOW_SIZE - 1:
        #         condition += " AND "
        rows = DbLogger.read_tuples_from_table(table_name=table_name)
        self.embeddingContextsAndTargets = np.zeros(shape=(len(rows), 2 * Constants.CBOW_WINDOW_SIZE + 1),
                                                    dtype=np.int32)
        context_arr = []
        print("Reading cbow data.")
        for i in range(len(rows)):
            row = rows[i]
            # label_ids = self.labelEncoder.transform(row)
            # context_arr.append(np.expand_dims(label_ids, axis=0))
            context_arr.append(np.expand_dims(np.array(row), axis=0))
            # for j in range(2 * Constants.CBOW_WINDOW_SIZE):
            #     self.embeddingContextsAndTargets[i, j] = row[j]
            # self.embeddingContextsAndTargets[i, -1] = row[-1]
        context_arr_2D = np.concatenate(context_arr, axis=0)
        self.encode_context_data(context_arr_2D=context_arr_2D)
        print("Reading completed. There are {0} contexts.".format(self.embeddingContextsAndTargets.shape[0]))

    def reset_training_state(self):
        self.currentIndices = np.array(range(self.embeddingContextsAndTargets.shape[0]))
        self.isNewEpoch = True
        np.random.shuffle(self.currentIndices)
        self.currentIndex = 0

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def get_next_batch(self, batch_size):
        num_of_samples = len(self.embeddingContextsAndTargets)
        curr_end_index = self.currentIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            curr_end_index = curr_end_index % num_of_samples
            indices_list = np.concatenate([indices_list, self.currentIndices[0:curr_end_index + 1]])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        self.currentIndex = self.currentIndex + batch_size
        context = self.embeddingContextsAndTargets[indices_list, 0:2 * Constants.CBOW_WINDOW_SIZE]
        targets = self.embeddingContextsAndTargets[indices_list, -1]
        unk_label = self.labelEncoder.transform(np.array(["UNK"]))[0]
        valid_entries = (context != unk_label).astype(np.float32)
        valid_entries_count = np.sum(valid_entries, axis=1)
        weights_per_row = np.reciprocal(valid_entries_count.astype(np.float32))
        weights = valid_entries * np.expand_dims(weights_per_row, axis=1)
        if num_of_samples <= self.currentIndex:
            self.reset_training_state()
        else:
            self.isNewEpoch = False
        return context, targets, weights
