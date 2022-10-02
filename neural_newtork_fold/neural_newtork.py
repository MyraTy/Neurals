import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_shortcuts as ts

print("Initialising...")
trainingI = ts.cdata([
    [[[[[1.0]]]], [[[[2.0]]]]],
    [[[[[3.0]]]], [[[[4.0]]]]]], type=float)
trainingO = ts.cdata([
    [[[[[1.0]]]], [[[[4.0]]]]],
    [[[[[9.0]]]], [[[[61.0]]]]]], type=float)
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
    res = ts.predict(seq, [float(input())], Type=float)
    print(res)
except ValueError:
    raise ValueError("That's not a float value. If you typed well, please ask MyraTy about this.")
except Exception:
    raise Exception("An error on source code ocured. Please ask MyraTy about this.")
finally:
    print("Program finished.")
