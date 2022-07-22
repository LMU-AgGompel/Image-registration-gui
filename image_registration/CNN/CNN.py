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
import itertools
import csv
import matplotlib.pyplot as plt
import pandas as pd

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
        for j in range(0,len(training.columns)):
            if type(training.iloc[i][j]) == str:
                training.iloc[i][j] = ast.literal_eval(training.iloc[i][j])
                training.iloc[i][j+1] = ast.literal_eval(training.iloc[i][j+1])
    
    b=[]
    c=[]
    for k in range(len(training.index)):
        for l in range(0,len(training.columns)):
            b += training.iloc[k][l]
        c.append(b)
        b=[]

    y2 = np.array(c)
    print(y2)
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=test_ratio, random_state=42)

    print(y_train)
    plt.imshow(X_train[0])
    return X_train, X_test, y_train, y_test

def create_CNN(X_train,y_train,X_test,y_test,model_folder, nb_epochs, img_shape, window, values, nb_batch_size= 16):
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
    img_shape : tuple, shape of images
    window :
    values : 
    nb_batch_size : int, number of images used per sub-iteration. Limited by the computer memory. The default is 16.

    Returns
    -------
    

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
    model.add(Dense(24))
    model.summary()

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    
    window['-MODEL-RUN-STATE-'].update('Yes', text_color=('lime'))
    window.Refresh()
    
    if values["-INF-EPOCHS-"] == True:
        for i in itertools.count(start=1):
            
            checkpoint = ModelCheckpoint(model_folder+ '/model.h5', monitor='mae', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
            
            window['-EPOCHS-COUNT-'].update('Epochs left : Inf')
            window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(model.history.history['mae']),2)))
            window.Refresh()
            
    else : 
        for i in range(nb_epochs):
            
            checkpoint = ModelCheckpoint(model_folder+ '/model.h5', monitor='mae', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)

            
            window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(nb_epochs - (+1)))
            window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(model.history.history['mae']),2)))
            window.Refresh()
            
    # Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    # here for 1000x1000 images : 256 000 000 000 / 4 / (1000*1000 + 256 822 328) = 248
    
    window['-MODEL-RUN-STATE-'].update('Saving model...', text_color=('yellow'))
    window.Refresh()

def continue_CNN(X_train,y_train,X_test,y_test,model_folder, nb_epochs, window, values, nb_batch_size= 16):
    '''
    Continue the training from a former model

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    model_folder : folder of the model to load
    nb_epochs : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.
    nb_batch_size : TYPE, optional
        DESCRIPTION. The default is 16.

    Returns
    -------
    None.

    '''
    continue_model = load_model(model_folder)
    
    window['-MODEL-RUN-STATE-'].update('Yes', text_color=('lime'))
    window.Refresh()
    
    if values["-INF-EPOCHS-"] == True:
        for i in itertools.count(start=1):
            checkpoint = ModelCheckpoint(model_folder, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            continue_model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)

            window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(nb_epochs - (i+1)))
            window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(continue_model.history.history['mae']),2)))
            window.Refresh()
             

    else : 
        # for i in range(nb_epochs):
            
        checkpoint = ModelCheckpoint(model_folder, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        continue_model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
        
        window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(nb_epochs - (i+1)))
        window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(continue_model.history.history['mae']),2)))
        window.Refresh()
            
    window['-MODEL-RUN-STATE-'].update('Saving model...', text_color=('yellow'))
    window.Refresh()
    
    
def predict_lm(df_files, df_model, values, window, shared):
    '''
    Create a csv file with the predicted coordinates of landmarks on images using the CNN model

    Parameters
    ----------
    df_files : TYPE
        DESCRIPTION.
    df_model : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    shared : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    window['-MODEL-RUN-STATE-'].update('Predicting...', text_color=('purple'))
    window.Refresh()
    
    folder_dir = values['-IMG-FOLDER3-']
    # get images 
    images_array = []
    os.chdir(folder_dir)

    #importing the images and recoloring them as grayscale
    for i in range(len(df_files["file name"])):
        images_array.append(color.rgb2gray(cv2.imread(df_files["file name"][i])))

    img = cv2.imread(df_files["file name"][0])
    X = np.asarray(images_array).reshape(len(df_files.index), img.shape[0],img.shape[1],1)


    model = load_model(values['-MODEL-FOLDER2-'])
    train_predicts = model.predict(X)
    
    

    prediction = []
    
    for j in range(len(df_files["file name"])):
        
        a = np.reshape(train_predicts[j],(1,len(df_model["target"])*2))
        a = np.reshape(a,(len(df_model["target"]),2))
        prediction.append(a)
        
    
    with open(shared['proj_folder'] + str('/predicted_landmarks_dataframe.csv'),'w',newline='') as file :

        write = csv.writer(file,delimiter = ',',quoting = csv.QUOTE_NONE, escapechar=' ')
        write.writerow(df_model["name"].values)
        write.writerows(prediction)

    window['-MODEL-RUN-STATE-'].update('No', text_color=('red'))
    window.Refresh()
        
    return 