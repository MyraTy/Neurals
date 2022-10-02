import tensorflow as tf
from tensorflow import keras
import numpy as np
import neurals_assets as na

print("Initialising...")
trainingI = na.cdata([
    [[[[[1.0]]]], [[[[2.0]]]]],
    [[[[[3.0]]]], [[[[4.0]]]]]], type=float)
trainingO = na.cdata([
    [[[[[1.0,]]]], [[[[11.0]]]]],
    [[[[[21.0]]]], [[[[1211.0]]]]]], type=float)
tf.random.set_seed(42)

print("Creating Neurals...")
seq = keras.Sequential([keras.layers.Dense(units=6, input_shape=[2, 1, 1, 1, 1], activation="relu")])
print("Compiling...")
seq.compile(optimizer=keras.optimizers.Adam(1), loss='mean_squared_error')

print("Training...")
seq.fit(trainingI, trainingO, epochs=7000)
print("Done!")
print("Enter 2 value to make Neurals predict it!")
print(na.predict(seq, [[[[[[float(input())]]]]], [[[[[float(input())]]]]]]))
