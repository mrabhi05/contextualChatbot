# Importing the Libraries 
from json import load
import nltk
from nltk.stem.lancaster import LancasterStemmer
from numpy.lib.function_base import append
import tensorflow
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import  tensorflow as tf
import  random
import json
import pickle

# Loading the JSON File 
with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","wb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = [] 
    docs_x = []    # List of all different patterns
    docs_y = []    # List of corresponding responses for patterns

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            letter = nltk.word_tokenize(pattern)
            words.extend(letter)
            docs_x.append(letter)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stemming words, lower casing it and storing it in list
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # Removes Duplicates
    words = sorted(list(set(words)))

    # Sorting the labels
    labels = sorted(labels)


    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        letter = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in letter:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as f: 
        pickle.dump((words, labels, training, output), f)

    
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearnchatbot")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearnchatbot")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for   word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)
    return np.array(bag)

def chat():
    print("Start talking with the bot[Type quit to Stop!] ")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]
        
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']            
            print(random.choice(responses))
        else:
            print("Sorry, I didn't get that, Try again.")

chat()
