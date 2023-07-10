import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import json
import pickle
import random

# loading intents.json

with open('intents.json') as intents_file:
    data = json.load(intents_file)

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

training = []
output = []

out_empty = [0 for _ in range (len(labels))]

# One hot encoding, Converting the words to numerals

for x, doc in enumerate(x_docs):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)


    output_row = out_empty[:]
    output_row[labels.index(y_docs[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Training the neural network

model = Sequential()
model.add(Dense(10, input_shape=(len(training[0]),), activation = 'relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(training, output, epochs= 500, batch_size =8, verbose=1)
model.save('my_model.keras')

# Making predictions

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(words.lower()) for word in s_words]

    for s_word in s_words:
        for i, w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    
    return np.array(bag)

# Creating a chat function that ties everything together

def chat():

    while True:
        inp = input("\n\nYou: ")
        if inp.lower() == 'quit':
            break

    # Probability of correct response

    results = model.predict([bag_of_words(inp, words)])

    # Picking the greatest number from probability

    results_index = np.argmax(results)

    tag = labels[results_index]

    for tg in data['intents']:

        if tg['tag'] == tag:
            responses = tg['responses']
            print("Bot: \t " + random.choice(responses))


chat()













