import numpy as np
import pandas as pd
import regex
from collections import Counter
from sklearn.preprocessing import LabelEncoder


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
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.vocabulary)
        print("X")