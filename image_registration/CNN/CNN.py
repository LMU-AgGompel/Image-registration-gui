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

def data_preprocessing_for_CNN(df_landmarks, df_files, df_model, test_ratio=0.3, augmented_data_folder = None, normalization = True):
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
    # Check if the augmented data folder exists and import data from there?
    
    if augmented_data_folder:
        try:
            training_data = pd.read_csv(os.path.join(augmented_data_folder, "augmented_landmarks.csv"))
            augmented_data_folder = os.path.join(augmented_data_folder, "")
            training_data["full path"] = augmented_data_folder+training_data["file name"]
            augmented = True
        except:
            training_data = df_landmarks.copy()
            training_data = training_data.dropna()
            training_data = training_data.reset_index()
            training_data = pd.merge(training_data, df_files, on=["file name"])
            augmented = False
        
    else:
    # Importing files
        training_data = df_landmarks.copy()
        training_data = training_data.dropna()
        training_data = training_data.reset_index()
        training_data = pd.merge(training_data, df_files, on=["file name"])
        augmented = False
        
    # get images and their landmarks
    images_array = []
    all_landmarks_positions = []
    n_files = len(training_data["full path"].unique())
    
    #importing the images and recoloring them as grayscale
    for image_path in training_data["full path"].unique():
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if normalization:
            mean = np.mean(image[image>0])
            image = image/mean
        
        images_array.append(image)
        landmark_positions = []
        
        for lmk in df_model["name"].unique():
            lmk_xy = ast.literal_eval(training_data.loc[training_data["full path"]==image_path, lmk].values[0])
            landmark_positions += lmk_xy
        
        all_landmarks_positions.append(landmark_positions)

    # reshape the images to be compatible with the neural network:
    # reshape has to be done with height, width:
 
    X = np.asarray(images_array, dtype = 'float').reshape(len(training_data["full path"].unique()), image.shape[0], image.shape[1], 1)
    y = np.array(all_landmarks_positions, dtype = 'float')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    return X_train, X_test, y_train, y_test, image.shape   


def create_CNN(img_shape, df_model):
    '''
    Function that will create a neural network from training data

    Parameters
    ----------
    img_shape : tuple, shape of images
    df_model  : pandas dataframe, contains the reference positions of the landmarks
        
    Returns
    -------
    model: the compiled tensorflow model for landmarks prediction
    
    '''

    n_landmarks = len(df_model["name"].unique())
    input_shape = (img_shape[0],img_shape[1],1)
    
    TF_FORCE_GPU_ALLOW_GROWTH=True

    model = Sequential()

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
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
    #model.add(Dropout(0.1))
    model.add(Dense(2*n_landmarks))
    model.summary()

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])
    
    return model

def load_CNN(model_path):
    landmarks_detection_model = load_model(model_path)
    return landmarks_detection_model

def train_CNN(X_train, y_train, X_test, y_test, landmarks_detection_model, model_path, nb_epochs, callbacks_list = [], nb_batch_size= 16):
    '''
    Continue the training from a former model

    Parameters
    ----------
    X_train : 
        Array of image training data
    X_test  : 
        Array of image testing data
    y_train : 
        Coordinates array of training data
    y_test  : 
        Coordinates array of testing data
    landmarks_detection_model: tensorflow model
        the compiled tensorflow model defined by create_CNN
    model_path : str
        path to save the current model
    nb_epochs : int
        number of training iterations.
    callbacks_list: list
        list of callbacks objects which defines which functions to run after each epoch
    nb_batch_size : int, optional
        Limited by computer memory. The default is 16.

    Returns
    -------
    None.

    '''
    
    landmarks_detection_model.fit(X_train, y_train, epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
    
    return 
    
def predict_lm(df_files, df_model, model, project_folder, lmk_filename = 'predicted_landmarks_dataframe.csv'):
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
    model : TYPE
        DESCRIPTION.
    project_folder : TYPE
        DESCRIPTION.
    lmk_filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    model_width = model.input_shape[2]
    model_height = model.input_shape[1]

    # get images 
    images = []

    # importing the images, converting to grayscale and resize
    for i in range(len(df_files["full path"])):
        image = color.rgb2gray(cv2.imread(df_files["full path"][i]))
        resized = cv2.resize(image, (model_width, model_height), interpolation = cv2.INTER_AREA)
        images.append(resized)

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    binning_w = image_width/model_width
    binning_h = image_height/model_height
    
    X = np.asarray(images).reshape(len(df_files["full path"].unique()), model_height, model_width, 1)
    train_predicts = model.predict(X)
    
    # reshape the predictions in an nd array with indexes corresponding to: file, landmark, coordinate.
    prediction = train_predicts[:,:,np.newaxis]
    prediction = prediction.reshape((len(df_files["file name"]), len(df_model["target"]), 2))
    
    # rescale the predicted landmarks positions to the original size of the image:
    prediction[:,:,0] = binning_w*prediction[:,:,0]
    prediction[:,:,1] = binning_h*prediction[:,:,1]
    
    # TO DO: reshape the results in the usual format of the landmark dataframe
    landmark_names = df_model['name'].unique()
    df_pred_lmk    = df_files[['file name']].copy()
    
    for landmark in landmark_names:
        df_pred_lmk[landmark] = np.nan
    
    for i in range(len(df_files["file name"])):
        for j in range(len(landmark_names)):
            lmk = landmark_names[j]
            x = int(prediction[i,j,0])
            y = int(prediction[i,j,1])
            df_pred_lmk.loc[df_pred_lmk ["file name"]==df_files["file name"][i], lmk] = str([x,y])
    
    df_pred_lmk.to_csv(os.path.join(project_folder, lmk_filename))

    return 