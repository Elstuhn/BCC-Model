import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import os
from visualize import *

earlystop = EarlyStopping(monitor = "val_accuracy", patience = 3)
model_save = ModelCheckpoint('BloodCells.hdf5',
                            save_best_only=True)

classes = ["Eosinophil", "Lymphocite", "Monocyte", "Neutrophil"]
labels = [0 for i in range(2497)] + [1 for i in range(2483)] + [2 for i in range(2478)] + [3 for i in range(2499)]
labelsY = [0 for i in range(623)] + [1 for i in range(620)] + [2 for i in range(620)] + [3 for i in range(624)]
labels = np.array(labels)
labelsY = np.array(labelsY)
trainX = []
testX = []
for files in os.listdir("TRAIN"):
    for image in os.listdir(f"TRAIN/{files}"):
        images = Image.open(f"TRAIN/{files}/{image}")
        images = images.resize((160, 120))
        images = np.array(images)
        trainX.append(images)
trainX = np.array(trainX)
trainX = trainX / 255.0
print(trainX.shape)

for files in os.listdir("TEST"):
    for image in os.listdir(f"TEST/{files}"):
        images = Image.open(f"TEST/{files}/{image}")
        images = images.resize((160, 120))
        images = np.array(images)
        testX.append(images)
testX = np.array(testX)
testX = testX / 255.0
print(testX.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (120, 160, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4, activation = 'softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
history = model.fit(trainX, labels, epochs=1000, validation_data = (testX, labelsY), callbacks = [earlystop, model_save], batch_size = 125)
score = model.evaluate(testX, labelsY, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
    
check_acc(model, testX, labelsY)
    
plothist(history, "accuracy")
plothist(history, "loss")
