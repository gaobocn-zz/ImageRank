# @Author: Gao Bo
# @Date:   2016-12-01T08:01:43-05:00
# @Last modified by:   Gao Bo
# @Last modified time: 2016-12-04T02:04:52-05:00



# load libraries
# %matplotlib inline
import numpy as np
from numpy.linalg import norm
import skimage.io as io
import matplotlib.pyplot as plt

import scipy.misc

import pylab

from sklearn.neighbors import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import time
import string
from collections import Counter
import random


class ImageRank(object):
    """docstring for image_rank."""
    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        # stop_words doesn't matter because tags don't include stop words
        # self.stop_words = set(stopwords.words('english'))
        # self.stemmer = nltk.stem.porter.PorterStemmer()

    def _download_nltk(self):
        """Call this the first time"""
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('words')

    # build dictionary based-on 2nd part of tags
    def build_dictionary(self, data_path='./tags_test/', start=0, end=2000):
        print('build_dictionary...')
        self.dictionary = {}
        self.category_dict = {}
        self.index2words = []
        word_i = 0
        for i in range(start, end):
            self.index2words.append([])
            for line in open(data_path + str(i) + '.txt'):
                tword = line.strip().split(':')[1]
                w_list = tword.split()
                # if len(w_list) > 1:
                    # w_list.append(''.join(w_list))
                for word in w_list:
                    word = self.wordnet_lemmatizer.lemmatize(word)
                    self.index2words[-1].append(word)
                    if word not in self.dictionary:
                        self.dictionary[word] = word_i
                        word_i += 1

    def get_cos_sim(self, vec1, vec2, delta=1):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + delta)

    def ensemble_cos_kNN(self, description_path='./descriptions_test/', start=0, end=2000):
        print('ensemble_cos_kNN...')
        cos_dist, cos_ind = self.cos_distance_classifier(description_path, start, end)
        # cos_dist, cos_ind = self.kNN_classifier(description_path, start, end)
        kNN_dist, kNN_ind = self.kNN_classifier(description_path, start, end)
        ind = []
        for i in range(start, end):
            tdist = kNN_dist[i]
            tdist = (tdist-min(tdist))/(max(tdist)-min(tdist))/4 - cos_dist[i]
            ind.append(tdist.argsort()[:20])
        return ind

    def cos_distance_classifier(self, description_path='./descriptions_test/', start=0, end=2000):
        print('cos_distance_classifier...')
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
                for word in line.strip().split('.')[0].split():
                    word = word.lower()
                    word = self.wordnet_lemmatizer.lemmatize(word)
                    if word in self.dictionary:
                        description_bow[i-start][self.dictionary[word]] = 1.0

        ind = []
        t_score = []
        for i in range(start, end):
            t_score.append([])
            for j in range(start, end):
                t_score[-1].append(self.get_cos_sim(description_bow[i-start], tag_bow[j-start]))
            ind.append(np.array(t_score[-1]).argsort()[-20:][::-1])

        return np.array(t_score), ind

    def kNN_classifier(self, description_path='./descriptions_test/', start=0, end=2000):
        print('kNN_classifier...')
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
                    w = self.wordnet_lemmatizer.lemmatize(w)
                    if w in self.dictionary:
                        description_bow[i-start][self.dictionary[w]] += 1.0

        # build KDTree for fast kNN
        kdt = KDTree(tag_bow)
        # kNN
        dist, ind = kdt.query(description_bow, k=end-start)
        for i in range(start, end):
            dist[i][ind[i]] = dist[i]
        return np.array(dist), ind[:,:20]

    def word_freq_classifier(self, description_path='./descriptions_test/', start=0, end=2000):
        print('word_freq_classifier...')
        # inverted_index {word: {description_index1: count1, description_index2: count2}}
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
                    w = self.wordnet_lemmatizer.lemmatize(w)
                    if w in self.inverted_index:
                        for j in self.inverted_index[w]:
                            if j in tdict:
                                tdict[j] += 1
                            else:
                                tdict[j] = 1
            tcounter = Counter(tdict).most_common(20)
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
    for i in range(len(ind)):
        f.write(str(i)+'.txt,')
        for j in range(len(ind[i])):
            if j == 19:
                f.write(str(ind[i][j])+'.jpg\n')
            else:
                f.write(str(ind[i][j])+'.jpg ')
    f.close()

def validation():
    data_len = 2000
    start = 0
    image_rank = ImageRank()
    image_rank.build_dictionary('./tags_train/', 0, data_len)
    # dist, ind = image_rank.kNN_classifier('./descriptions_train/', 0, data_len)
    # dist, ind = image_rank.cos_distance_classifier('./descriptions_train/', 0, data_len)
    ind = image_rank.ensemble_cos_kNN('./descriptions_train/', 0, data_len)
    # print(dist[:10][:10])
    score = 0
    for i in range(data_len):
        for j in range(20):
            if ind[i][j] == start + i:
                score += 20 - j
    score = (score / 20.0) / data_len
    print('Score is: ' + str(score))
    # write_submission(ind)


if __name__ == '__main__':
    run_validation = False
    start = time.time()

    if run_validation:
        validation()
    else:
        image_rank = ImageRank()
        image_rank.build_dictionary()
        ind = image_rank.ensemble_cos_kNN()
        write_submission(ind)

    end = time.time()
    print('Total time elapsed: ' + str(end - start))
