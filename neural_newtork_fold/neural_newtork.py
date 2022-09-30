from tensorflow import keras as k
import numpy as np
import synteticer
print("Creando red neuronal...")
m = k.models.Sequential()
print("Hello, this is a neural newtork.", "First, you have to enter the training data, that are the training inputs and outputs respectivly. When you enter it, you must do it like a Python list, please.", "Let's begin!", sep="\n")
rInput = input()
rOutput = input()
Input = rInput.split(", ")
Output = rOutput.split(", ")
count = 0
for s1 in Input:
    Input[count] = int(s1)
    count += 1
count = 0
for s2 in Output:
    Output[count] = int(s2)
    count += 1
a = ""
b = ""
trainingI = np.array(Input, "float32")
trainingO = np.array(Output, "float32")
m.add(k.layers.Dense(1, input_dim=1, activation="softmax"))
m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
print("Training...")
print("Epochs?")
e = int(input())
m.fit(trainingI, trainingO, epochs=e)
print("Ready!")
scores = m.evaluate(trainingI, trainingO)
print("%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
print("Type a value for make yor neural neutork predict that!")
print(m.predict(synteticer.synt(int(input()))))
print("And that's all!!!")
