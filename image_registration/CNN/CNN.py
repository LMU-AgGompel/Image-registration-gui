#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:35:49 2022

@author: titouan
"""

from skimage import color
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Convolution2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os,cv2,ast
import numpy as np

def initialize_CNN(shared, path, df_landmarks, df_files, df_model, img_shape, test_ratio=0.3):
    '''
    Function to initalize the data for the neural network

    Parameters
    ----------
    shared : TYPE
        DESCRIPTION.
    df_landmarks : TYPE
        DESCRIPTION.
    df_files : TYPE
        DESCRIPTION.
    test_ratio : int, Ratio of data that will be used for testing the neural network. The default is 0.3.

    Returns
    -------
    X_train : X coordinates array of training data
    X_test : X coordinates array of testing data
    y_train : y coordinates array of training data
    y_test : y coordinates array of testing data

    '''
    # Importing files
    training = df_landmarks
    training = training.dropna()
    training = training.reset_index()
    folder_dir = path
    
    
    landmarks_list = df_model["name"].values

    
    # get images and their landmarks
    images_array = []
    
    os.chdir(folder_dir)
    
    #importing the images and recoloring them as grayscale
    for i in range(len(training["file name"])):
        images_array.append(color.rgb2gray(cv2.imread(training["file name"][i])))


    # Removing images with empty values
    training = training.drop(["file name"], axis=1)
    training = training.drop(['index'],axis = 1)


    X = np.asarray(images_array).reshape(len(training.index),img_shape[1],img_shape[0],1)

    for i in range(len(training.index)):
        for j in range(0,len(training.columns)-1,2):
            training.iloc[i][j] = ast.literal_eval(training.iloc[i][j])
            training.iloc[i][j+1] = ast.literal_eval(training.iloc[i][j+1])
    
    b=[]
    c=[]
    for k in range(len(training.index)):
        for l in range(0,len(training.columns)-1,2):
            b += training.iloc[k][l]
        c.append(b)
        b=[]

    y2 = np.array(c)
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=test_ratio, random_state=42)


    return X_train, X_test, y_train, y_test

def create_CNN(X_train,y_train,X_test,y_test,model_folder, nb_epochs, img_shape, nb_batch_size= 16):
    '''
    Function that will create a neural network from training data

    Parameters
    ----------
    X_train : Array of iamge training data
    X_test : Array of image testing data
    y_train : Coordinates array of training data
    y_test : Coordinates array of testing data
    model_folder : str, folder where to save the model
    nb_epochs : int, number of training iterations
    nb_batch_size : int, number of images used per sub-iteration. Limited by the computer memory. The default is 16.

    Returns
    -------
    score : array containing different precision values of the model

    '''
    TF_FORCE_GPU_ALLOW_GROWTH=True

    model = Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape = img_shape[::-1] + (1,) ))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))       
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12))
    model.summary()

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    history = model.fit(X_train, y_train, epochs = nb_epochs, batch_size = nb_batch_size)
            
    # Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    # here for 1000x1000 images : 256 000 000 000 / 4 / (1000*1000 + 256 822 328) = 248, we round it up to 256 for it be a power of 2
    # cant handle that much, 100 works

    training_loss = history.history['loss']
    training_mae = history.history['mae']
    
    score = model.evaluate(X_test, y_test, verbose=0)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    model.summary()
    model.save(model_folder+ '/model.h5')
    # model2 = load_model(model_folder + '/model.h5')
    # model2.summary()
    return score

def continue_CNN(X_train,y_train,X_test,y_test,model_path, nb_epochs, nb_batch_size= 16):

    continue_model = load_model(model_path)
    continue_model.summary()
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    continue_model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
    score = continue_model.evaluate(X_test, y_test, verbose=0)
    return score
