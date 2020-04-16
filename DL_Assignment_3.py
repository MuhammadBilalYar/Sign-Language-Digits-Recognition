# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:46:49 2020

@author: Muhammad.Bilal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping

#Load Images
X = np.load("X.npy")
#Data is 2062 x 64 x 64 matrix
print(X.shape)
#Load labels - One hot encoded: 2062 x 10
Y = np.load("Y.npy")
print(Y.shape)

#sample_image = X[0,:,:]
#plt.subplot(221)
#plt.imshow(sample_image, cmap="gray")
#sample_image = X[500,:,:]
#plt.subplot(222)
#plt.imshow(sample_image, cmap="gray")
#sample_image = X[1000,:,:]
#plt.subplot(223)
#plt.imshow(sample_image, cmap="gray")
#sample_image = X[2000,:,:]
#plt.subplot(224)
#plt.imshow(sample_image, cmap="gray")


# Train Test split
test_size = 0.20
number_of_classes=10
epochs=100

X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
X_train, X_test, Y_train, Y_test = train_test_split(X_conv, Y, stratify=Y, test_size=test_size, random_state=42)

# 
def Evaluate_CNN_Model(model, modelName, optimizer=optimizers.RMSprop(lr=0.0001), callbacks=None):
    print("[INFO]:Convolutional Model {} created...".format(modelName))    
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print("[INFO]:Convolutional Model {} compiled...".format(modelName))
    
    print("[INFO]:Convolutional Model {} training....".format(modelName))
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience=20, verbose = 1) 
    if callbacks is None:
        callbacks = [earlyStopping]

    print("[INFO]:Convolutional Model {} trained....".format(modelName))

    test_scores=model.evaluate(X_test, Y_test, verbose=0)
    train_scores=model.evaluate(X_train, Y_train, verbose=0)
    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))
    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))

    return model


def build_conv_model_1():
    model=Sequential()
    
    model.add(layers.Conv2D(64, kernel_size=(3,3),
                           padding="same",
                           activation="relu", 
                           input_shape=(64, 64,1)))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(number_of_classes, activation="softmax"))
        
    return model

model=build_conv_model_1()
trained_model_1=Evaluate_CNN_Model(model=model, modelName=1)