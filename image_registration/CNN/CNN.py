"""
Created on Wed Jul 13 12:35:49 2022

@authors: stefano, titouan
"""

from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, MaxPool2D, Convolution2D, LeakyReLU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os, ast
import numpy as np
import pandas as pd
import math
import random as rd
from scipy import ndimage
import glob
import threading
import copy
import PIL

def image_downsample(img, binning, normalize=True):
    """
    Parameters
    ----------
    img : 2D numpy array
        image to downsample.
    binning : int
        binning factor.
    normalize: bool, default = True
        whether to normalize the image after binning.
        
    Returns
    -------
    img : 2D numpy array
        binned image.
    """
    width = int(img.shape[0] / binning)
    height = int(img.shape[1] / binning)
    dim = (width, height)
    img = resize(img, dim, preserve_range=True, anti_aliasing=True).astype('uint16')
    
    if normalize:
        img = img / np.mean(img[img>0])
        
    return img

def landmarks_downsample(file_name, df_landmarks, df_model, binning):
    """
    Parameters
    ----------
    file_name : str
        name of the current image.
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.
    binning : int
        binning factor.

    Returns
    -------
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.

    """
    
    for lmk in df_model["name"].unique():
    
        x,y = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"]==file_name, lmk].values[0])
        x = int(x/binning)
        y = int(y/binning)
        df_landmarks.loc[df_landmarks["file name"]==file_name, lmk] = str([x,y])
    
    return df_landmarks

def rotate_image(image, angle):
    """

    Parameters
    ----------
    image : 2D numpy array
        image.
    angle : float
        rotation angle in degrees.

    Returns
    -------
    rotated_image : 2D numpy array
        rotated image.

    """
    av_color = np.mean(image [image >0])
    rotated_image = ndimage.rotate(image, -angle, reshape = False,  mode='constant', cval=av_color)
    return rotated_image

def scale_image(image, zoom):
    """
    Zoom in or out in an image, and returns an output image of the same size of 
    the input.
    If the original image size is reduced, the missing values at the boundaries 
    are filled in with the average image brightness.
    
    Parameters
    ----------
    image : 2D numpy array
        image.
    zoom : float
        zoom factor, if >1 the image is magnified.

    Returns
    -------
    2D numpy array
        rescaled image.

    """
    av_color = np.mean(image [image >0])
    if zoom >1:
        scaled_image = ndimage.zoom(image, zoom, mode='constant', cval=av_color)
    
        o_y_s, o_x_s = image.shape
        n_y_s, n_x_s = scaled_image.shape
        
        startx = n_x_s//2 - o_x_s//2
        starty = n_y_s//2 - o_y_s//2
    
        new_image = scaled_image[starty:starty+o_y_s, startx:startx+o_x_s]
        return new_image
    else:
        scaled_image = ndimage.zoom(image, zoom,  mode='constant', cval=av_color)
        new_image = np.ones(image.shape)*av_color
        
        o_y_s, o_x_s = image.shape
        n_y_s, n_x_s = scaled_image.shape
        
        startx = o_x_s//2 - n_x_s//2
        starty = o_y_s//2 - n_y_s//2
        
        new_image[starty:starty+n_y_s, startx:startx+n_x_s] = scaled_image
        
        return new_image
        
        

def duplicate_landmarks(df_landmarks, df_model, new_filename, old_filename):
    """
    Add a new row to the landmarks dataframe corresponding to a new filename
    and fills the landmarks positions by duplicating those of an existing image.
    
    Parameters
    ----------
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.
    new_filename : TYPE
        DESCRIPTION.
    old_filename : TYPE
        DESCRIPTION.

    Returns
    -------
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.

    """
    ## add a new row to the landmarks_train dataframe:
    df_landmarks.loc[len(df_landmarks.index), "file name"] = new_filename
    
    for lmk in df_model["name"].unique():
        lmk_values = df_landmarks.loc[df_landmarks["file name"] == old_filename, lmk].values[0]
        df_landmarks.loc[df_landmarks["file name"] == new_filename, lmk] = lmk_values
        
    return df_landmarks

def scale_landmarks(img, zoom, df_landmarks, df_model, filename):
    """
    Update the landmark positions of the image specified by filename, by 
    rescaling their distances form the image center.
    
    Parameters
    ----------
    img : 2D numpy array
        current image.
    zoom : float
        zoom factor.
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.
    filename : str
        current filename.

    Returns
    -------
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.

    """
    ox, oy = img.shape[1]/2, img.shape[0]/2
    
    for lmk in df_model["name"].unique():

        x,y = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"] == filename, lmk].values[0])

        qx = round(ox + zoom * (x - ox) )
        qy = round(oy + zoom * (y - oy) )
        
        df_landmarks.loc[df_landmarks["file name"] == filename, lmk] = str([qx,qy])
    
    return df_landmarks

def rotate_landmarks(img, angle, df_landmarks_train, df_model, filename):
    """
    Update the landmark positions of the image specified by filename, by 
    rotating them around the image center.
    

    Parameters
    ----------
    img : 2D numpy array
        current image.
    angle: float
        rotation angle, in degrees.
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.
    filename : str
        current filename.

    Returns
    -------
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.

    """
    ox, oy = img.shape[1]/2, img.shape[0]/2

    rad_angle = angle*math.pi/180
    
    for lmk in df_model["name"].unique():

        x,y = ast.literal_eval(df_landmarks_train.loc[df_landmarks_train["file name"] == filename, lmk].values[0])

        qx = round(ox + math.cos(rad_angle) * (x - ox) - math.sin(rad_angle) * (y - oy))
        qy = round(oy + math.sin(rad_angle) * (x - ox) + math.cos(rad_angle) * (y - oy))
        df_landmarks_train.loc[df_landmarks_train["file name"] == filename, lmk] = str([qx,qy])
    
    return df_landmarks_train


def training_data_preprocessing(output_folder_train, output_folder_val, df_landmarks, df_files, df_model, n_data_augmentation, binning, test_size=0.2, normalization=True):
    
    # Creating the folder for the training and validation datasets,
    # which contain the images and the corresponding landmarks files.
    
    try : 
        os.mkdir(output_folder_train) 
        os.mkdir(output_folder_val) 
    except : 
        print('removing files from training and validation folder')
        files = glob.glob(os.path.join(output_folder_train,"*"))
        for f in files:
            os.remove(f)
        files = glob.glob(os.path.join(output_folder_val,"*"))
        for f in files:
            os.remove(f)
    
    # clean up the data to remove files with missing landmarks from training:
    df_landmarks_copy = df_landmarks.copy()
    df_landmarks_copy = df_landmarks_copy.dropna()
    df_landmarks_copy = df_landmarks_copy.reset_index(drop = True)
    
    # separate images in training and validation sets:
    df_landmarks_val = df_landmarks_copy.sample(int(test_size*len(df_landmarks_copy)), ignore_index=True).reset_index(drop = True)
    df_landmarks_train = df_landmarks_copy.drop(df_landmarks_val.index).reset_index(drop = True)

    # bin, normalize and augment the training data:
    for file_name in df_landmarks_train["file name"].unique():
        
        img = PIL.Image.open(df_files.loc[df_files["file name"] == file_name, "full path"].values[0])
        img = np.asarray(img)
        
        if binning:
            img = image_downsample(img, binning, normalization)
            df_landmarks_train = landmarks_downsample(file_name, df_landmarks_train, df_model, binning)
            
        # save the image in the training folder:
        img_PIL = PIL.Image.fromarray(img)
        img_PIL.save(os.path.join(output_folder_train, file_name))

        # randomly rotate the image and its corresponding landmarks:  
        for k in range(n_data_augmentation):
            angle  = rd.randint(0, 360)
            zoom   = rd.randint(700, 1300)/1000.0
            new_file_name = str(k)+"_"+file_name
            rotated_image = rotate_image(img, angle)
            scaled_image  = scale_image(rotated_image, zoom)
            #add noise to the image:
            gauss_noise = np.random.normal(0, 0.02, scaled_image.shape)
            augmented_image = scaled_image + gauss_noise
            augmented_image_PIL = PIL.Image.fromarray(augmented_image)
            augmented_image_PIL.save(os.path.join(output_folder_train, new_file_name))
            # duplicate the landmarks for the current file:
            df_landmarks_train = duplicate_landmarks(df_landmarks_train, df_model, new_file_name, file_name)
            # rotate the landmarks of the new file:
            df_landmarks_train = rotate_landmarks(augmented_image, angle, df_landmarks_train, df_model, new_file_name)
            # scale the landmarks for the new file:
            df_landmarks_train = scale_landmarks(augmented_image , zoom, df_landmarks_train, df_model, new_file_name)

    
    print("Finished augmenting training data.")
    
    # bin and normalize the validation data:
    for file_name in df_landmarks_val["file name"].unique():
        img = PIL.Image.open(df_files.loc[df_files["file name"] == file_name, "full path"].values[0])
        img = np.asarray(img)
        
        if binning:
            img = image_downsample(img, binning, normalization)
            df_landmarks_val = landmarks_downsample(file_name, df_landmarks_val, df_model, binning)
                
        # save the image in the validation folder:
        img_PIL = PIL.Image.fromarray(img)
        img_PIL.save(os.path.join(output_folder_val, file_name))
            
    df_landmarks_train.to_csv(os.path.join(output_folder_train, "df_landmarks.csv"))
    df_landmarks_val.to_csv(os.path.join(output_folder_val, "df_landmarks.csv"))
    
    return


def import_data(folder, df_landmarks, df_files, df_model):
    """
    Import the images and landmark positions from a folder and reshape the data
    to return a n_pixels*n_samples feature vector and n_landmarks*n_samples array 
    with all the landmark positions.

    Parameters
    ----------
    folder : str
        path to the images.
    df_landmarks : pandas dataframe
        dataframe of landmarks positions.
    df_files : pandas dataframe
        dataframe of image files info. Including the the full path to each image.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.

    Returns
    -------
    X : 2D numpy array
        reshaped images.
    y : 2D numpy array
        reshaped landmark positions.
        
    or
    
    None, if the images in the project don't have the same size

    """
    
    df_landmarks["full path"] = os.path.join(folder,"")+df_landmarks["file name"]
    
    images_array = []
    all_landmarks_positions = []
    # check the shape of the images:
    img_shape = check_image_shape(df_files)
    
    if img_shape:
        #importing the images converting them to grayscale and append to a list:
        for image_path in df_files["full path"].unique():
            
            image = PIL.Image.open(image_path)
            image = np.asarray(image)
            images_array.append(image)
            landmark_positions = []
            
            for lmk in df_model["name"].unique():
                lmk_xy = ast.literal_eval(df_landmarks.loc[df_landmarks["full path"]==image_path, lmk].values[0])
                landmark_positions += lmk_xy
            
            all_landmarks_positions.append(landmark_positions)
            
        # reshape the images to be compatible with the neural network, reshape with height, width:        
        X = np.asarray(images_array, dtype = 'float').reshape(len(df_landmarks["full path"].unique()), img_shape[0], img_shape[1], 1)
        y = np.array(all_landmarks_positions, dtype = 'float')
        return X, y
    
    else:
        return None

        
def import_train_val_data(trainig_data_folder, validation_data_folder, df_model):
    """
    Read the training data and validation data folders and create the reshaped 
    training and validation X and y arrays.

    Parameters
    ----------
    trainig_data_folder : str
        path to the (augmented) training data.
    validation_data_folder : str
        path to the validation data.
    df_model : pandas dataframe
        dataframe with landmark names and target positions.

    Returns
    -------
    X_train : 2D numpy array
        reshaped training images.
    X_val : 2D numpy array
        reshaped validation images.
    y_train : 2D numpy array
        reshaped training landmark positions.
    y_val : 2D numpy array
        reshaped validation landmark positions.

    """

    df_landmarks_train = pd.read_csv( os.path.join(trainig_data_folder,"df_landmarks.csv") )
    df_landmarks_val   = pd.read_csv( os.path.join(validation_data_folder,"df_landmarks.csv") )

    X_train, y_train = import_data(trainig_data_folder, df_landmarks_train, df_landmarks_train, df_model)
    X_val, y_val = import_data(validation_data_folder, df_landmarks_val, df_landmarks_val, df_model)

    return X_train, X_val, y_train, y_val

def check_image_shape(df_files):
    """
    Check if all the images in the current project have the same size and 
    return the size. If the size is not the same for all the images then 
    returns None.

    Parameters
    ----------
    df_files : TYPE
        DESCRIPTION.

    Returns
    -------
    img_shape : TYPE
        DESCRIPTION.

    """
    
    img_path = df_files["full path"].unique()[0]
    img = PIL.Image.open(img_path)
    img = np.asarray(img)
    img_shape = img.shape
    
    for img_path in df_files["full path"].unique():
        img = PIL.Image.open(img_path)
        img = np.asarray(img)
        if img_shape != img.shape:
            return None

    return img_shape


def create_CNN(img_shape, df_model):
    '''
    Function that will create the convolutional neural network structure for a 
    given shape of input images.

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


def train_CNN_with_window_callback(train_folder, val_folder, proj_folder, df_model, nb_epochs, model_name, CNN_model_object, window):
    """
    Import the training and validation data and train the CNN model in a new 
    thread, using a custom callback that interacts with the GUI window.
    
    Parameters
    ----------
    train_folder : str
        DESCRIPTION.
    val_folder : str
        DESCRIPTION.
    proj_folder : str
        DESCRIPTION.
    df_model : pandas dataframe
        DESCRIPTION.
    nb_epochs : int
        DESCRIPTION.
    model_name : srt
        DESCRIPTION.
    CNN_model_object : keras model
        DESCRIPTION.
    window : PySimpleGUI window
        DESCRIPTION.

    Returns
    -------
    None.

    """
    X_train, X_test, y_train, y_test = import_train_val_data(train_folder, val_folder, df_model)
    callbacks_list = [window_callback(window, nb_epochs, proj_folder, model_name = model_name)]
    
    threading.Thread(target = train_CNN,
                     args = (X_train, y_train, X_test, y_test, CNN_model_object, 
                             proj_folder, nb_epochs, callbacks_list), 
                     daemon=True).start()    
    return

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
    
    landmarks_detection_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
    
    return

def fine_tune_CNN_with_window_callback(train_folder, val_folder, proj_folder, df_model, nb_epochs, model_name, CNN_model_object, window):
    
    X_train, X_test, y_train, y_test = import_train_val_data(train_folder, val_folder, df_model)
    
    callbacks_list = [window_callback(window, nb_epochs, proj_folder, model_name = model_name)]
    
    threading.Thread(target = fine_tune_CNN,
                     args = (X_train, y_train, X_test, y_test, CNN_model_object, 
                             proj_folder, nb_epochs, callbacks_list), 
                     daemon=True).start()  
    return

def fine_tune_CNN(X_train, y_train, X_test, y_test, CNN_model, model_path, nb_epochs, callbacks_list = [], nb_batch_size= 16):
    
    # Make BatchNormalization layers non trainable
    for layer in CNN_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            
    # Recompile the model and use a smaller step for training:
    CNN_model.compile(optimizer = Adam(1e-5), loss ='mse', metrics = ['mae'])
    CNN_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epochs, batch_size = nb_batch_size, callbacks=callbacks_list)
    
    return
    
def predict_lm(df_files, df_model, model, project_folder, normalization=True, lmk_filename = 'predicted_landmarks_dataframe.csv'):
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

    # get images 
    images = []
    model_width = model.input_shape[2]
    model_height = model.input_shape[1]

    # importing the images, converting to grayscale and resize
    for image_name in df_files["full path"].unique():

        image = PIL.Image.open(image_name)
        image = np.asarray(image)
        
        if normalization:
            image = image / np.mean(image[image>0])
            
        resized_image = resize(image, (model_height, model_width), preserve_range=True, anti_aliasing=True)
        #resized = cv2.resize(image, (model_width, model_height), interpolation = cv2.INTER_AREA)
        images.append(resized_image)

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
    
    # reshape the results in the usual format of the landmark dataframe
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


class window_callback(Callback):
    """
    Customized Callback class to update the gui window during training and save
    the model every n epochs.
    """
    def __init__(self, window, nb_epochs, folder, save_freq=10, model_name = "landmarks_detection_model.h5"):
        super(window_callback, self).__init__()
        self.window = window
        self.epochs_left = nb_epochs
        self.folder = folder
        self.save_freq = save_freq
        self.model_name = model_name
        self.filepath = os.path.join(self.folder, self.model_name)+".h5"
        
    def on_train_begin(self, logs={}):
        self.metrics = {}
        self.previous_metrics = {}
        self.current_epoch = 0
        for metric in logs:
            self.metrics[metric] = []
            self.previous_metrics[metric] = []
        self.window['-MODEL-RUN-STATE-'].update('Yes', text_color=('lime'))
        self.window.Refresh()

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        self.previous_metrics = copy.deepcopy(self.metrics)
        for metric in logs:
            self.metrics[metric] = logs.get(metric)

        self.window['-EPOCHS-COUNT-'].update('Epochs left : ' + str(self.epochs_left))
        self.epochs_left = self.epochs_left-1
        self.current_epoch = self.current_epoch+1
        
        if 'mae' in self.metrics:
            self.window['-CURRENT-MAE-'].update('Current training precision (mae): ' + str(round(self.metrics['mae'],2)))    
        elif 'mean_absolute_error' in self.metrics:
            self.window['-CURRENT-MAE-'].update('Current training precision (mae): ' + str(round(self.metrics['mean_absolute_error'],2)))
        else:
            self.window['-CURRENT-MAE-'].update('Current training precision (mse): ' + str(round(self.metrics['loss'],2)))

        if 'val_mae' in self.metrics:
            self.window['-CURRENT-VALMAE-'].update('Current validation precision (val_mae): ' + str(round(self.metrics['val_mae'],2)))    
        elif 'val_mean_absolute_error' in self.metrics:
            self.window['-CURRENT-VALMAE-'].update('Current validation precision (val_mae): ' + str(round(self.metrics['val_mean_absolute_error'],2)))
        else:
            self.window['-CURRENT-VALMAE-'].update('Current validation precision (val_mse): ' + str(round(self.metrics['val_loss'],2)))

        self.window.Refresh()
        
        # store the model at the end of the epoch only if the validation loss is reduced:
        if self.current_epoch > 1:
            if self.metrics['val_loss'] < self.previous_metrics['val_loss']:
                self.model_to_save = copy.deepcopy(self.model)
            # save the best model of the last save_freq epochs
            if self.epochs_left % self.save_freq == 0:
                self.model_to_save.save(self.filepath, overwrite=True)
        else:    
            self.model_to_save = copy.deepcopy(self.model)
            
    def on_train_end(self, logs={}):
        self.window['-MODEL-RUN-STATE-'].update('No', text_color=('red'))
        self.window.Refresh()
        filepath = os.path.join(self.folder, self.model_name)+".h5"
        self.model.save(filepath, overwrite=True)