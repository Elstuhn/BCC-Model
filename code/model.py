import tensorflow as tf
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import RandomizedSearchCV
from visualize import *
from data-clean import *

def makemodel():
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
  return model
