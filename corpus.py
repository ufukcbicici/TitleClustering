import numpy as np
import pandas as pd
import re


class Corpus:
    FILTER_REGEX = "[^a-zA-Z\d\s\-,，\\/]|[\r\n]"

    def __init__(self):
        self.clearedTitles = None

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
                words = re.split("\s", title[:-1])
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
        for i in range(titles_df.shape[0]):
            raw_title = titles_df.loc[i][0]
            cleared_title = re.sub(Corpus.FILTER_REGEX, "", raw_title)
            words = re.split("[\\-,，\s]", cleared_title)
            words = [word.capitalize() for word in words if word != ""]
            self.clearedTitles.append(np.array(words))
            print("Title {0}: {1}".format(i, self.clearedTitles[-1]))
        self.clearedTitles = np.array(self.clearedTitles)