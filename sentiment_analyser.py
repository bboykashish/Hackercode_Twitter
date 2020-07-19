import nltk
import random #random sampling without replacement
import pickle
from nltk.classify import ClassifierI #standard interface for “single-category classification”, in which the set of categories is known, the number of categories is finite, and each text belongs to exactly one category.
from nltk.tokenize import word_tokenize
from statistics import mode #function that returns the mode 
from nltk.collocations import BigramCollocationFinder #collocations are pairs of words frequently occuring together. Bigrams are sets of 2 adjacent words
from nltk.metrics import BigramAssocMeasures  #The nltk.metrics package provides a variety of evaluation measures which can be used for a wide variety of NLP tasks.
 #A collection of bigram association measures



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers): #classifiers contains the databses i.e the pos and neg
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

l_d = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(l_d)
l_d.close()

w_features = open("pickled_algos/word_features.pickle", "rb")
word_features = pickle.load(w_features)
w_features.close()

def find(documents):
    temp = {}
    word = set(documents)
    n = 2
    score_fn=BigramAssocMeasures.chi_sq

    bigram_finder = BigramCollocationFinder.from_words(word)
    bigrams = bigram_finder.nbest(score_fn, n)
    
    for w in word_features:
        temp[w] = (w in word )
    for w in bigrams:
        temp[w] = True
        
    return temp

classify_buffer = open("pickled_algos/classifier.pickle", "rb")
classifier = pickle.load(classify_buffer)
classify_buffer.close()

def sentiment(text):
    feat = find(text)
    return VoteClassifier(classifier).classify(feat),VoteClassifier(classifier).confidence(feat)
