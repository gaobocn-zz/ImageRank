# load libraries
# %matplotlib inline
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.misc
import pylab
from sklearn.neighbors import KDTree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time
from collections import Counter
import random


class ImageRank(object):
    """docstring for image_rank."""
    def __init__(self, data_path='./tags_test/', start=0, end=2000, load_data=False):
        # initialize wordnet lemmatizer (need to download WordNet Corpus from NLTK)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        # English stop words list
        self.stop_words = set(stopwords.words('english'))
        self.dictionary, self.index2words = self.build_dictionary(data_path, start, end)
        self.test_n = end - start


    # build dictionary based-on 2nd part of tags
    def build_dictionary(self, data_path='./tags_test/', start=0, end=2000):
        self.dictionary = {}
        self.index2words = []
        word_i = 0
        for i in range(start, end):
            self.index2words.append([])
            for line in open(data_path + str(i) + '.txt'):
                w = line.strip().split(':')[1]
                w = w.split()
                for word in w:
                    word = self.wordnet_lemmatizer.lemmatize(word)
                    self.index2words[-1].append(word)
                    if word not in self.dictionary:
                        self.dictionary[word] = word_i
                        word_i += 1
        return self.dictionary, self.index2words


    def kNN_baseline(self, description_path='./descriptions_test/', start=0, end=2000):
        # binary BoW representation of test tags
        tag_bow = np.zeros((end - start, len(self.dictionary)), dtype=np.float32)
        for i in range(start, end):
            for word in self.index2words[i-start]:
                if word in self.dictionary:
                    tag_bow[i-start][self.dictionary[word]] = 1.0


        # binary BoW representation of test descriptions
        description_bow = np.zeros((end - start, len(self.dictionary)), dtype=np.float32)
        for i in range(start, end):
            for line in open(description_path + str(i) + '.txt'):
                for w in line.strip().split('.')[0].split():
                    w = w.lower()
                    if w not in self.stop_words:
                        w = self.wordnet_lemmatizer.lemmatize(w)
                        if w in self.dictionary:
                            description_bow[i-start][self.dictionary[w]] = 1.0

        # build KDTree for fast kNN
        kdt = KDTree(tag_bow)

        # kNN
        dist, ind = kdt.query(description_bow, k=20)

        return ind


    def build_inverted_index(self, description_path='./descriptions_test/', start=0, end=2000):
        # inverted_index { word : { description_index1 : count1, description_index2 : count2 } }
        self.inverted_index = {}
        for word in self.dictionary:
            self.inverted_index[word] = {}

        ind = []
        for i, word_list in enumerate(self.index2words):
            for word in word_list:
                if word in self.dictionary:
                    if i in self.inverted_index[word]:
                        self.inverted_index[word][i] += 1
                    else:
                        self.inverted_index[word][i] = 1

        for i in range(start, end):
            tdict = {}
            for line in open(description_path + str(i) + '.txt'):
                for w in line.strip().split('.')[0].split():
                    w = w.lower()
                    if w not in self.stop_words:
                        w = self.wordnet_lemmatizer.lemmatize(w)
                        if w in self.inverted_index:
                            for j in self.inverted_index[w]:
                                if j in tdict:
                                    tdict[j] += 1
                                else:
                                    tdict[j] = 1
            tcounter = Counter(tdict).most_common(20)
            if i == 0:
                print(tcounter)
            ind.append([])
            for key, val in tcounter:
                ind[-1].append(key)
            while len(ind[-1]) < 20:
                rand_i = random.randint(0, end - start - 1)
                if rand_i not in tdict:
                    ind[-1].append(rand_i)

        return ind


def write_submission(ind):
    # output rank list on test set
    f = open('bow_baseline.csv', 'w')
    f.write('Descritpion_ID,Top_20_Image_IDs\n')
    for i in range(2000):
        f.write(str(i)+'.txt,')
        for j in range(20):
            if j == 19:
                f.write(str(ind[i][j])+'.jpg\n')
            else:
                f.write(str(ind[i][j])+'.jpg ')
    f.close()

def validation():
    data_len = 5000
    image_rank = ImageRank('./tags_train/', 0, data_len)
    # ind = image_rank.kNN_baseline('./descriptions_train/', 0, data_len)
    ind = image_rank.build_inverted_index('./descriptions_train/', 0, data_len)
    score = 0
    for i in range(data_len):
        for j in range(20):
            if ind[i][j] == i:
                score += 20 - j
    score = (score / 20) / data_len
    print('Score is: ' + str(score))
    # write_submission(ind)


if __name__ == '__main__':
    run_test = True
    start = time.time()

    if run_test:
        validation()
    else:
        image_rank = ImageRank()
        # ind = image_rank.kNN_baseline()
        ind = image_rank.build_inverted_index()
        write_submission(ind)


    end = time.time()
    print('Total time elapsed: ' + str(end - start))
