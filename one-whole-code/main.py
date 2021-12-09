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
import matplotlib.pyplot as plt
from PIL import Image
import os

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

def check_acc(model, X_test, y_test):
    incorrect = 0
    correct = 0
    for i in range(len(X_test)):
        prediction = model.predict(X_test[i].reshape(1, 120, 160, 1)).argmax()
        actual = y_test[i]
        if int(prediction) != int(actual):
            incorrect += 1
        else:
            correct += 1
            
    fig = plt.figure()
    plt.ylabel("Amount")
    plt.xlabel(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    plt.bar(["correct", "incorrect"], [correct, incorrect])
    plt.show()
    print(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    
#check_acc(model, testX, labelsY)

def plothist(history, metric):
    plt.figure()
    if metric == "accuracy":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel("Accuracy")
    elif metric == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
    
plothist(history, "accuracy")
plothist(history, "loss")
