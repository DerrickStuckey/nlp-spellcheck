# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 09:59:06 2015

@author: dstuckey

Addition to Peter Norvig's basic spellcheck program using bigram frequency
instead of unigram frequency for candidate selection
"""

import nltk
import norvig_spellcheck as ns
from functools import partial

#calculate bigram frequency distribution
bgs = nltk.bigrams(ns.words(file('big.txt').read()))
fdist = nltk.FreqDist(bgs)

#basic bigram frequency
def word_probability_mle(prevWord, word):
    bigramCount = fdist[(prevWord, word)]
    prevWordCount = ns.NWORDS[prevWord]
    return (bigramCount + 0.0) / prevWordCount

#makes use of bigram frequency
def bigram_correct(prevWord, word):
    candidates = ns.known([word]) or ns.known(ns.edits1(word)) or ns.known_edits2(word) or [word]
    scoreFunc = partial(word_probability_mle, prevWord)
    return max(candidates, key=scoreFunc)