#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# geolocate.py : Naive Bayes classifier to predict the location of tweet.
# Augustine Joseph, Mandar Sudhir Baxi, Milind
# Course: CS-B551-Fall2017
#
##################################################################################################
#                                                                                                #
#                                                                                                #
#       A  multinomial Naive Bayes Model has been built to classify tweets.                      #
#                                                                                                #
##################################################################################################
#
#
#
"""
#
# Assumptions and Strategies :
#   1.  The original input data contained several non-ascii characters. These were removed using linux 
#       shell commands.
#   2.  The code is written to accept only the data cleaned in step 1: 
#            tweets.train.clean.txt and tweets.test1.clean.txt 
#   3.  The code implements the following cleaning processes:
#           Remove punctuation, remove empty strings, change the case to lower, remove stop words
#   4.  The stop words list created based on NLTK's list of English stopwords
#   5.  We chose multinomial Naive Bayes Model rather than Bernoulli model
#   6.  Comparitive study of both models has shown that the multinomial model is usually superior to the Bernoulli model.
#   7.  Reference: McCallum, A., Nigam, K.: A Comparison of Event Models for Naive Bayes Text Classiﬁcation. 
#       In: AAAI/ICML-98 Workshop on Learning for Text Categorization. (1998) 41–48
#   8.  Code execution format: python ./geolocate.py training-file testing-file output-file
#   9.  Initial results showed an accuracy of 65.8%
#   10. To improve results words common to all categories were collected in a list and removed.
#   11. This improved the accuracy to 67%
#   12. The probability P(W|L) has been caculated during the training process. This word with highest probability
#       has been captured in a OrderedDict.
#   13. Below is a list of 5 top words from each category:
#           Top 5 words for the location  Atlanta,_GA are:      ['atlanta', 'ga', 'georgia', 'atl', 'night']
#           Top 5 words for the location  Boston,_MA are:       ['boston', 'ma', 'report', 'massachusetts', 'fenway']
#           Top 5 words for the location  Chicago,_IL are:      ['chicago', 'il', 'illinois', 'want', 'night']
#           Top 5 words for the location  Houston,_TX are:      ['houston', 'tx', 'texas', 'nursing', 'healthcare']
#           Top 5 words for the location  Los_Angeles,_CA are:  ['ca', 'los', 'angeles', 'losangeles', 'hollywood']
#           Top 5 words for the location  Manhattan,_NY are:    ['new', 'york', 'ny', 'nyc', 'newyork']
#           Top 5 words for the location  Orlando,_FL are:      ['orlpol', 'orlando', 'opd', 'fl', 'ave']
#           Top 5 words for the location  Philadelphia,_PA are: ['philadelphia', 'pa', 'philly', 'pennsylvania', 'phillies']
#           Top 5 words for the location  San_Diego,_CA are:    ['ca', 'san', 'sandiego', 'diego', 'california']
#           Top 5 words for the location  San_Francisco,_CA are:['ca', 'san', 'sanfrancisco', 'francisco', 'california']
#           Top 5 words for the location  Toronto,_Ontario are: ['toronto', 'ontario', 'trucks', 'bw', '2']
#           Top 5 words for the location  Washington,_DC are:   ['dc', 'washington', 'day', 'district', 'washingtondc']
#
#   14. The ouput will be written to a specified file 
#   15. There are situations where the testing tweet has words that are part of the vocabulary but not in training set
#       for a category. Laplace smoothing has been applied to care for the zero probability situations.
#   16. Words not in vocabulary, but present in test data were ignored. 
#    
#
"""


from __future__ import print_function
import string
import numpy as np
import operator
import itertools
import re
import codecs
import sys, math, time
from collections import Counter, OrderedDict


class tweets:
    def __init__(self, type):

        self.type = type
        self.tweet_words = []
        self.class_type_list = []
        self.n_tweets = 0
        self.class_type = []
        self.class_freq = []
        self.class_prob = []
        self.class_words = []
        self.class_words_freq = []
        self.vocabulary = []
        self.count_vocab = 0
        self.count_class_words = []
        self.word_prob = []
        self.post_prob = []

    def cleanup(self,w_list):

        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                      'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                      'was', 'were', 'be', 'been', 'being', 'have','has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                      'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                      'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                      'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some','such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

        all_common = ['job', 'hiring', 'im', 'jobs', 'careerarc', 'latest', 'amp', 'opening', 'click', 'st', 'work', 'see', 'great']

        w_list = [''.join(c for c in s if c not in string.punctuation) for s in w_list]  # Remove punctuation
        w_list = [s for s in w_list if s]  # remove empty strings
        w_list = [word.lower() for word in w_list]  # change the case to lower
        w_list = [w for w in w_list if w not in stop_words]  # remove stop words
        w_list = [w for w in w_list if w not in all_common]  # remove words common to all locations
        return w_list


    def read_data(self, fname):

        tweets_data = open(fname, 'r').read().split('\n')

        td = []

        for i in range(0, len(tweets_data)):
        #for i in range(0, 20):
            if re.search(r"(,_..)", tweets_data[i]):
                if i > 0:
                    self.tweet_words.append(td)
                td = tweets_data[i].split()
                self.class_type_list.append(td[0])
                td.pop(0)  # removing the location information

                td = self.cleanup(td) # not for stan data
                #print(td)
            else:

                td1 = tweets_data[i].split()
                td1 = self.cleanup(td1) # not for stan data
                td = td + td1

        self.tweet_words.append(td)
        self.n_tweets = len(self.tweet_words)
        #self.class_type, self.class_freq = np.unique(self.class_type_list, return_counts=True) #numpy1.9 only

        class_counter = Counter(self.class_type_list)
        class_counter = OrderedDict(sorted(class_counter.items(), key=lambda t: t[0]))
        self.class_type = list(class_counter.keys())
        self.class_freq = list(class_counter.values())

        # Calculate P(c), probability of each class (location)

        self.class_prob = [x / float(self.n_tweets) for x in self.class_freq]

        print("Number of tweets: ", self.n_tweets)
        print("Number of classes: ", len(self.class_type))


    def words_by_class(self):

        ######Create list of words by class

        for i in range(0, len(self.class_type)):
            class_type_words = []
            for j in range(0, len(self.class_type_list)):
                if self.class_type[i] == self.class_type_list[j]:
                    class_type_words = class_type_words + self.tweet_words[j]
            #cl_words, cl_words_freq = np.unique(class_type_words, return_counts=True) # numpy 1.9 only

            word_counter = Counter(class_type_words)
            Top_counter = OrderedDict(word_counter.most_common())
            top_words = list(Top_counter.keys())
            if self.type == 'train':
                print("Top 5 words for the location ", self.class_type[i], "are:")
                print(top_words[:5])
                #print("\n")
            word_counter = OrderedDict(sorted(word_counter.items(), key=lambda t: t[0]))
            cl_words = list(word_counter.keys())
            cl_words_freq = list(word_counter.values())

            self.class_words.append(cl_words)
            self.class_words_freq.append(cl_words_freq)

    #######Create vocabulary

    def create_vocabulary(self):

        vocab_words = []
        for i in range(0, len(self.class_type)):
            vocab_words.extend(self.class_words[i])

        self.vocabulary = np.unique(vocab_words)
        self.count_vocab = len(self.vocabulary)



    # Calculate the total number of words in each class

    def count_words_by_class(self):

        for i in range(0, len(self.class_type)):
            count_cl_words = np.sum(self.class_words_freq[i])

            self.count_class_words.append(count_cl_words)



######### Train Naive Bayes

    def train_nb(self):

        for i in range(0, len(self.class_type)):
            w_0_prob = 1/float(self.count_class_words[i] + self.count_vocab) # Probability of a word not in class type, but in test data.
            w_prob = [(x + 1.0)* w_0_prob for x in self.class_words_freq[i]]  # Applying the Laplace Smoothing
            #w_prob = np.array(self.class_words_freq[i] + 1) / float(self.count_class_words[i] + self.count_vocab) #  numpy 1.9

            w_prob = np.append(w_prob, w_0_prob) # store w_0_prob for retrieval from the end of the list

            self.word_prob.append(w_prob)



    ##### Test Naive Bayes

    def test_nb(self, train_set, out_file):
        #Ignored all words in test that are not in vocabulary
        predicted = []
        orig = []
        file = open(out_file, 'w')
        for k in range(0, len(self.class_type_list)):
        #for k in range(0, 1):
            test_words = self.tweet_words[k]

            test_words = [w for w in test_words if w in train_tweet.vocabulary] # remove words not in vocabulary

            self.post_prob = []
            for i in range(0, len(train_set.class_type)):
                c_prob = 1 # Conditional probability

                for j in range(0, len(test_words)):
                    test_word = test_words[j]
                    if test_word in train_set.class_words[i]:
                        word_idx = train_set.class_words[i].index(test_word)

                        c_prob *= train_set.word_prob[i][word_idx]
                        # print(c_prob)
                    else:
                        c_prob *= train_set.word_prob[i][-1] # effect of Laplace smoothing on words in vocabulary and test data but not in class
                c_prob *= train_set.class_prob[i]

                self.post_prob.append(c_prob)

            index, value = max(enumerate(self.post_prob), key=operator.itemgetter(1))

            predicted.append(train_set.class_type[index])
            orig.append(self.class_type_list[k])

            file.write(str(train_set.class_type[index]) + ' ' + str(self.class_type_list[k]) + ' ' + ' '.join(self.tweet_words[k]) + '\n') # for txt file

        true_positive = len([i for i, j in zip(predicted, orig) if i == j]) / float( self.n_tweets)
        print("True Positive: ", true_positive)

args = sys.argv
train_file = args[1]
test_file = args[2]
outfile = args[3]
start = time.time()
train_tweet = tweets(type='train')


train_tweet.read_data(fname=train_file)
train_tweet.words_by_class()
train_tweet.create_vocabulary()
train_tweet.count_words_by_class()
train_tweet.train_nb()

test_tweet = tweets(type='test')


test_tweet.read_data(fname=test_file)
test_tweet.words_by_class()

test_tweet.test_nb(train_set=train_tweet, out_file=outfile)
print('Execution time: %0.2fs' % (time.time() - start))
