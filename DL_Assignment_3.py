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
    history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks,epochs=epochs, verbose=0)
    print("[INFO]:Convolutional Model {} trained....".format(modelName))

    test_scores=model.evaluate(X_test, Y_test, verbose=0)
    train_scores=model.evaluate(X_train, Y_train, verbose=0)
    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))
    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))
    
    Show_Model_History(history, modelName)
    return model

def Show_Model_History(modelHistory, modelName):
    history=pd.DataFrame()
    history["Train Loss"]=modelHistory.history['loss']
    history["Validation Loss"]=modelHistory.history['val_loss']
    history["Train Accuracy"]=modelHistory.history['accuracy']
    history["Validation Accuracy"]=modelHistory.history['val_accuracy']
    
    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(12,8))
    axarr[0].set_title("History of Loss in Train and Validation Datasets")
    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])
    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")
    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1]) 
    plt.suptitle(" Convulutional Model {} Loss and Accuracy in Train and Validation Datasets".format(modelName))
    plt.show()

def Build_Conv_Model(filters,filterSize):
    model = Sequential()
    model.add(layers.Convolution2D(filters, filterSize, activation='relu', padding="same", input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
       
    model.add(layers.Convolution2D((filters*2), filterSize, activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Convolution2D((filters*2), filterSize, activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Convolution2D((filters*4), filterSize, activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    
        
    model.add(layers.Flatten())
    
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
      
    return model

print("=========================================================")
print("\t Experiment with fiter size of 3 * 3")
print("=========================================================")
filters = 32
filterSize = (3,3)
model=Build_Conv_Model(filters,filterSize)
trained_model_1=Evaluate_CNN_Model(model=model, modelName=1)

print("=========================================================")
print("\t Experiment with fiter size of 5 * 5")
print("=========================================================")
filters = 32
filterSize = (5,5)
model=Build_Conv_Model(filters,filterSize)
trained_model_1=Evaluate_CNN_Model(model=model, modelName=2)

print("=========================================================")
print("\t Experiment with fiter size of 7 * 7")
print("=========================================================")
filterSize = (7,7)
model=Build_Conv_Model(filters,filterSize)
trained_model_1=Evaluate_CNN_Model(model=model, modelName=3)