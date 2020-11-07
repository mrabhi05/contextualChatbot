# Libraries needed for NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


# Libraries needed for Tensorflow processing 
import tensorflow as tf
import numpy as np
import tflearn
import random
import json

# Import our chat-bot intents file
with open('intents.json') as json_data:
    intent = json.load(json_data)


