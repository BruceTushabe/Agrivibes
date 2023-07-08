import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle
import random



# loading intents,json

with open('intents,json') as intents:
    data = json.load(intents)

stemmer = LancasterStemmer()

# getting information from intents.json

words = []
labels = []
x_docs = []
y_docs = []

for intent in data ['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        x_docs.append(wrds)
        y_docs.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Stemming the words and removing duplicate elements.

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)







