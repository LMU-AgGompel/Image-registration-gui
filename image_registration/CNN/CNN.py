"""
Created on Wed Jul 13 12:35:49 2022

@authors: titouan, stefano
"""

from skimage import color
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Convolution2D
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os,cv2,ast
import numpy as np
import itertools
import csv
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


    X = np.asarray(images_array).reshape(len(training.index), img_shape[1], img_shape[0], 1)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=test_ratio, random_state=42)

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
    
    # 
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
            model.fit(X_train, y_train, epochs=1, batch_size = nb_batch_size, callbacks=callbacks_list)

            
            window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(nb_epochs - (i+1)))
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

            window['-EPOCHS-COUNT-'].update('Epochs left : Inf')
            window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(continue_model.history.history['mae']),2)))
            window.Refresh()
             

    else : 
        for i in range(nb_epochs):
            
            checkpoint = ModelCheckpoint(model_folder, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            continue_model.fit(X_train, y_train, epochs=1, batch_size = nb_batch_size, callbacks=callbacks_list)
        
            window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(nb_epochs - (i+1)))
            window['-CURRENT-MAE-'].update('Current precision : ' + str(round(min(continue_model.history.history['mae']),2)))
            window.Refresh()
            
    window['-MODEL-RUN-STATE-'].update('Saving model...', text_color=('yellow'))
    window.Refresh()
    
    
def predict_lm(df_files, df_model, values, window, shared):
    '''
    Create a csv file with the predicted coordinates of landmarks on images using the CNN model.
    It also rescales the images to match the input size of the selected model and rescales back
    the positions of the predicted landmarks.

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

    # load the model and get width and height to which the images have to be rescaled:
    try:
        model = load_model(values['-MODEL-FOLDER2-'])
        model_width = model.input_shape[2]
        model_height = model.input_shape[1]
    
        # get images 
        images = []
    
        # importing the images, converting to grayscale and resize
        for i in range(len(df_files["full path"])):
            image = color.rgb2gray(cv2.imread(df_files["full path"][i]))
            resized = cv2.resize(image, (model_width, model_height))
            images.append(resized)
    
        image_width = image.shape[0]
        image_height = image.shape[1]
        
        binning_w = image_width/model_width
        binning_h = image_height/model_height
        
        X = np.asarray(images).reshape(len(df_files["full path"].unique()), model_height, model_width, 1)
        train_predicts = model.predict(X)
        
        # reshape the predictions in an nd array with indexes corresponding to: file, landmark, coordinate.
        prediction = train_predicts[:,:,np.newaxis]
        prediction = prediction.reshape((len(df_files["file name"]), len(df_model["target"]), 2))
        
        # rescale the predicted landmarks positions to the original size of the image:
        prediction[:,:,0] = binning_h*prediction[:,:,0]
        prediction[:,:,1] = binning_w*prediction[:,:,1]
        
        # TO DO: reshape the results in the usual format of the landmark dataframe
        landmark_names = df_model['name'].values
        df_pred_lmk    = df_files[['file name']].copy()
        
        for landmark in landmark_names:
            df_pred_lmk[landmark] = np.nan
        
        for i in range(len(df_files["file name"])):
            for j in range(len(landmark_names)):
                lmk = landmark_names[j]
                x = int(prediction[i,j,0])
                y = int(prediction[i,j,1])
                df_pred_lmk.loc[df_pred_lmk ["file name"]==df_files["file name"][i], lmk] = str([x,y])
        
        df_pred_lmk.to_csv(os.path.join(shared['proj_folder'], 'predicted_landmarks_dataframe.csv'))
    
    except:
        window["-PRINT-"].update("An error occured during landmarks prediction.")
        
    window['-MODEL-RUN-STATE-'].update('No', text_color=('red'))
    window.Refresh()

    return 