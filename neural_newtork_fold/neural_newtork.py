import random
import json
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_shortcuts as ts

import nltk
from nltk.stem import WorldNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore = [".", ",", "!", "?"]

print("Initialising...")

for intent in intents["Intents"]

"""trainingI = ts.cdata([
    [[[[[codify("Hello world!")]]]], [[[[codify("I'm a terminator")]]]]],
    [[[[[codify("I'm not your father")]]]], [[[[codify("My home")]]]]]], type=float)
trainingO = ts.cdata([
    [[[[[codify("No film")]]]], [[[[codify("film")]]]]],
    [[[[[codify("film")]]]], [[[[codify("film")]]]]]], type=float)
tf.random.set_seed(42)

print("Creating Neurals...")
seq = keras.Sequential([keras.layers.Dense(units=6, input_shape=[2, 1, 1, 1, 1], activation="relu")])
print("Compiling...")
seq.compile(optimizer=keras.optimizers.Adam(1.5), loss='mean_squared_error')

print("Training...")
seq.fit(trainingI, trainingO, epochs=1000, verbose=False)
print("Done!")
print("Enter a float value to make Neurals predict it!")
try:
    res = ts.predict(seq, [codify(input())], Type=float)
    count = len(str(res))
    print(decodify(res, count))
except ValueError:
    raise ValueError("That's not a float value. If you typed well, please ask MyraTy about this.")
except Exception:
    raise Exception("An error on source code ocured. Please ask MyraTy about this.")
finally:
    print("Program finished.")"""
