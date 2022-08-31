"""
Created on Wed Jul 13 12:35:49 2022

@authors: titouan, stefano
"""

from skimage import color
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, MaxPool2D, Convolution2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os,cv2,ast
import numpy as np
import itertools
import pandas as pd

def data_preprocessing_for_CNN(df_landmarks, df_files, df_model, new_width, new_height, test_ratio=0.3):
    '''
    Function to preprocess the data for the neural network

    Parameters
    ----------
    df_landmarks : pandas dataframe
        DESCRIPTION.
    df_files     : pandas dataframe
        DESCRIPTION.
    df_model     : pandas dataframe
        DESCRIPTION.
    test_ratio   : int, Ratio of data that will be used for testing the neural network. The default is 0.3.

    Returns
    -------
    X_train : X coordinates array of training data
    X_test  : X coordinates array of testing data
    y_train : y coordinates array of training data
    y_test  : y coordinates array of testing data

    '''
    # TO DO: Check if augmented data folder exists and import data from there?
    
    # Importing files
    training_data = df_landmarks.copy()
    training_data = training_data.dropna()
    training_data = training_data.reset_index()
    training_data = pd.merge(training_data, df_files, on=["file name"])

    # get images and their landmarks
    images_array = []
    all_landmarks_positions = []
    
    #importing the images and recoloring them as grayscale
    for image_path in training_data["full path"].unique():
        image = color.rgb2gray(cv2.imread( image_path ))
        img_shape = image.shape
        binning_x, binning_y = img_shape[0]/new_width, img_shape[1]/new_height 
        image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
        images_array.append(image)
        
        landmark_positions = []
        
        for lmk in df_model["name"].unique():
            lmk_xy = ast.literal_eval(training_data.loc[training_data["full path"]==image_path, lmk].values[0])
            lmk_xy = [lmk_xy[0]/binning_x, lmk_xy[1]/binning_y]
            landmark_positions += lmk_xy
        
        all_landmarks_positions.append(landmark_positions)
    
    img_shape = (new_width, new_height)
    
    # reshape the images to be compatible with the neural network:
    X = np.asarray(images_array).reshape(len(training_data.index), img_shape[1], img_shape[0], 1)
    y = np.array(all_landmarks_positions)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    return X_train, X_test, y_train, y_test

def create_CNN(X_train, y_train, X_test, y_test, img_shape, df_model):
    '''
    Function that will create a neural network from training data

    Parameters
    ----------
    X_train : Array of image training data
    X_test  : Array of image testing data
    y_train : Coordinates array of training data
    y_test  : Coordinates array of testing data
    model_folder : str, folder where to save the model
    nb_epochs    : int, number of training iterations
    img_shape    : tuple, shape of images
    df_model:
    window :
    values : 
    nb_batch_size : int, number of images used per sub-iteration. Limited by the computer memory. The default is 16.

    Returns
    -------
    

    '''
    TF_FORCE_GPU_ALLOW_GROWTH=True
    
    n_landmarks = len(df_model["name"].unique())

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

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3,3), padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3,3), padding='same',use_bias=False))
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
    model.add(Dense(2*n_landmarks))
    
    #model.summary()
    
    # 
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    
    return model

def load_CNN(model_path):
    landmarks_detection_model = load_model(model_path)
    return landmarks_detection_model

def train_CNN(X_train, y_train, X_test, y_test, landmarks_detection_model, model_path, nb_epochs, callbacks_list = [], nb_batch_size= 16):
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
    landmarks_detection_model: tensorflow model
    
    model_path : str
        path to save the current model
    nb_epochs : TYPE
        DESCRIPTION.
    callbacks_list: list of callbacks objects
    
    nb_batch_size : TYPE, optional
        The default is 16.

    Returns
    -------
    None.

    '''
    
    ## Has to return the model?

    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list.append(checkpoint)
    landmarks_detection_model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
    return 
    
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
            resized = cv2.resize(image, (model_width, model_height), interpolation = cv2.INTER_AREA)
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