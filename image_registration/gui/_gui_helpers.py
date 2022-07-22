import io
import os
import shutil
import glob
import math
import PySimpleGUI as sg
from PIL import Image
import pandas as pd
import numpy as np
import ast
from datetime import datetime
import cv2
from ..registration.TPS import TPSwarping
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.filters import difference_of_gaussians
import matplotlib.pyplot as plt
import random as rd
import image_registration

file_types_dfs = [("CSV (*.csv)", "*.csv"),("All files (*.*)", "*.*")]

file_types_images = [("tiff (*.tiff)", "*.tif"), ("All files (*.*)", "*.*")]

image_quality_choiches = ["undefined", "good", "fair", "poor", "bad"]

df_files_name = "images_dataframe.csv"

df_landmarks_name = "landmarks_dataframe.csv"

df_lines_name = "lines_dataframe.csv"

df_model_name = "model_dataframe.csv"

ref_image_name = "reference_image.tif"




def update_image_fields(im_index, image, df_files, window, graph_canvas_width):
    """
    Function used to update all the fields related to the current image 
    when the image is changed:
    
    Parameters
    ----------
    im_index : int
        index of the current image.
    image : numpy array
    
    df_files : pandas Dataframe
    
    window : PySimplegui window
    
    graph_canvas_width: int
        width of the graph element of the main window

    Returns
    -------
    None.

    """

    filename = df_files.loc[im_index,"file name"]
    image_quality = df_files.loc[im_index,"image quality"]
    image_notes = df_files.loc[im_index,"notes"]
    image_annotated = df_files.loc[im_index,"annotated"]
    window["-CURRENT-IMAGE-"].update(value=filename)
    window["-IMAGE-QUALITY-"].update(value=image_quality)
    window["-IMAGE-NOTES-"].update(value=image_notes)
    window["-IMAGE-ANNOTATED-"].update(value=image_annotated)
    update_image(image, window, graph_canvas_width)
    
    return


def update_image(image, window, canvas_width):
    """
    Function used to show an image on the graph element of a window
    
    Parameters
    ----------
    image : numpy array
        image to show on the window graph element.
    window : PySimplegui window
        window with a graph object.
    canvas_width : int
        width of the graph object

    Returns
    -------
    None.

    """
    
    width = image.width
    height = image.height
    scaling_factor = canvas_width/width
    new_width = canvas_width
    new_height = int(height*scaling_factor)
    
    image = image.resize((new_width, new_height))

    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window['-GRAPH-'].erase()
    window['-GRAPH-'].set_size((new_width, new_height))
    window['-GRAPH-'].change_coordinates((0,0), (width, height), )
    window['-GRAPH-'].draw_image(data=bio.getvalue(), location=(0,height))
    return

def open_image(image_path, normalize=True):
    """
    Opens the image file, converts it into 8 bit, and optionally, normalizes it 
    
    Parameters
    ----------
    image_path : str
        path to the image.
    normalize : bool, optional.

    Returns
    -------
    a Pillow image.

    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = convert_image_to8bit(image, normalize)
    image = Image.fromarray(image)
    return image

def update_landmarks_preview(image_path, window, canvas_width, normalize=True):
    """
    Function used to update the image in the landmarks preview element of the main window.
    It opens the reference image file, converts it into 8 bit, normalizes it and resizes it.
    
    Parameters
    ----------
    image_path : str
        path to the image.
    window : PySimplegui window
        main window with the -LANDMARK-PREVIEW- element.
    canvas_width : int
        width of the -LANDMARK-PREVIEW- graph element.
    normalize : bool, optional
        wether the image should be normalized before visualization. The default is True.

    Returns
    -------
    None.

    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = convert_image_to8bit(image, normalize)
    image = Image.fromarray(image)
    
    width = image.width
    height = image.height
    scaling_factor = canvas_width/width
    new_width = canvas_width
    new_height = int(height*scaling_factor)
    
    image = image.resize((new_width, new_height))

    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window['-LANDMARKS-PREVIEW-'].erase()
    window['-LANDMARKS-PREVIEW-'].set_size((new_width, new_height))
    window['-LANDMARKS-PREVIEW-'].change_coordinates((0,0), (width, height), )
    window['-LANDMARKS-PREVIEW-'].draw_image(data=bio.getvalue(), location=(0,height))
    return

def refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, main_window, landmarks_window):
    """
    Parameters
    ----------
    shared : dictionary
        contains data shared across windows, definition is in the main function.
    df_files : pandas DataFrame
        dataframe containing the paths to images of current project.
    df_model : pandas DataFrame
        dataframe containing the names and positions of landmarks of current project.
    df_landmarks : pandas DataFrame
        dataframe containing the positions of landmarks and additional notes for
        all images of current project.
    main_window : PySimplegui window
        main window of the GUI.
    landmarks_window : PySimplegui window
        landmark selection window of the GUI.

    Returns
    -------
    shared : dictionary
        updated data shared across windows.
    landmarks_window: PySimplegui window
        resfreshed landmark selection window of the GUI.
    """
    
    # updated current image, raw_image and current file:
    shared['curr_image'] = open_image(df_files.loc[shared['im_index'],"full path"], normalize=shared['normalize'])
    shared['raw_image'] = shared['curr_image']
    shared['curr_file'] = df_files.loc[shared['im_index'],"file name"]
    shared['pt_size'] = shared['curr_image'].width / 80
    
    # update all the fields related to the image (image qualty, notes, etc..)
    update_image_fields(shared['im_index'], shared['curr_image'], df_files, main_window, shared['graph_width'])
    
    # refresh the landmarks window, if present
    if landmarks_window:
        location = landmarks_window.CurrentLocation()
        temp_window = make_landmarks_window(df_model, df_landmarks, shared['curr_file'], location = location)
        landmarks_window.Close()
        landmarks_window = temp_window
        
    # else, create a new one:
    else:
        landmarks_window = make_landmarks_window(df_model, df_landmarks, shared['curr_file'])
    
    # update the preview of the landmarks: 
    update_landmarks_preview(os.path.join(shared['proj_folder'], ref_image_name), main_window, 300)
    
    # remove selection of the current landmark
    shared['curr_landmark'] = None
    
    # update the progress bar
    update_progress_bar(df_files, main_window)
    
    # remove focus from any object
    try:
        x = main_window.FindElementWithFocus()
        x.block_focus()
    except:
        pass
    
    # place the focus on the main window:
    main_window.TKroot.focus_force() 
    
    return shared, landmarks_window


def convert_image_to8bit(image, normalize=False):
    """
    Helper function to convert an image in 8 bit format and, optionally, normalize it
    
    Parameters
    ----------
    image : numpy array
    normalize : bool, optional
        wether the image shoudl be normalized. The default is False.

    Returns
    -------
    image : numpy array

    """
    if normalize:
        image = image - np.min(image)
        image = image/np.max(image)
        image = 255*image
    else:
        max_val = np.iinfo(image.dtype).max
        image = image/max_val
        image = 255*image
        
    image = np.uint8(image)
    return image

def convert_image_coordinates_to_graph(x, y, im_width, im_height):
    """
    
    Function used to convert image coordinates in graph coordinates.
    The y coordinate in a PySimplegui Graph element is inverted compared to 
    the usual definition for images.
    
    Parameters
    ----------
    x : int
    y : int
    im_width : int
        width of the image.
    im_height : int
        height of the image.

    Returns
    -------
    [x,y] list
        graph coordinates.

    """
    y = im_height - y
    return [x, y]

def convert_graph_coordinates_to_image(x, y,im_width, im_height):
    """
    Function used to convert graph coordinates in image coordinates.
    The y coordinate in a PySimplegui Graph element is inverted compared to 
    the usual definition for images.
    
    Parameters
    ----------
    x : int
    y : int
    im_width : int
        width of the image.
    im_height : int
        height of the image.

    Returns
    -------
    [x,y] list
        image coordinates.

    """
    y = im_height - y
    return [x, y]
 
def update_progress_bar(df_files, window):
    """
    Function used to update the progress bar in the main window of the GUI 
    and print a message that shows how many images have been annotated.

    Parameters
    ----------
    df_files : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n_not_annotated = len(df_files[df_files['annotated']=="No"].index)
    n_annotated = len(df_files[df_files['annotated']=="Yes"].index)
    progress = 100*(n_annotated/(n_annotated+n_not_annotated))
    window["-PROGRESS-"].update(current_count = progress)
    window["-PRINT-"].update(str(n_annotated)+" annotated images out of "+str(n_not_annotated+n_annotated) )
    return   

def rebin(img, binning):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width = int(img.shape[1] / binning)
    height = int(img.shape[0] / binning)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def enhance_edges(img, binning, smoothing):
    img = rebin(img, binning)
    filtered_image = difference_of_gaussians(img, 2, 10)
    filtered_image = (-filtered_image)*(filtered_image<0).astype(np.uint8)
    filtered_image = 1 -filtered_image
    edges = (filtered_image-np.min(filtered_image))/(np.max(filtered_image)-np.min(filtered_image))
    edges = gaussian(edges, smoothing, preserve_range=False)
    return edges

#Issue with point spacing : if point spacing > lenght between the two points -> error
def snake_contour(img, p1_x, p1_y, p2_x, p2_y, alpha, smoothing, w_line, N = None, points_spacing = 30, binning=2 ):
    distance = np.sqrt((p1_x-p2_x)**2+(p1_y-p2_y)**2)
    n_points = N or int(distance/points_spacing)

    r = np.linspace(p1_y, p2_y, n_points)/binning
    c = np.linspace(p1_x, p2_x, n_points)/binning
        
    init = np.array([r, c]).T
    img = enhance_edges(img, binning, smoothing)
    snake = active_contour(img,
                   init, boundary_condition='fixed', coordinates='rc', 
                   alpha=0.1, beta=1, w_line=-5, w_edge=0, gamma= 0.1)
    return snake*binning
    

def create_new_project():
    """
    Function used to create a new project.
    It opens a new graphical window and collects from the user the info required
    for the creation of a new project: 
        
        - the project location
        - a folder of raw images
        - a reference image
        - a model file containig the definitions of the landmarks to be used by
          the project
          
    Finally, it creates all the new project files in the target folder.
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [[sg.Text("Project name: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-NEW-PROJECT-NAME-')],
              
              [sg.Text('Project location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-PROJECT-FOLDER-'),
               sg.FolderBrowse()],
              
              [sg.Text("Image files extension: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-IMAGE-EXTENSION-')],
              
              [sg.Text('Raw images location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-IMAGES-FOLDER-'),
               sg.FolderBrowse()], 
              
              [sg.Text("Reference image: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key="-NEW-REF-IMAGE-"),
               sg.FileBrowse(file_types=file_types_images)], 
              
              [sg.Text("Model file: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key="-NEW-MODEL-FILE-"),
               sg.FileBrowse(file_types=file_types_dfs)],
              
              [sg.Button("Create the project: ", size = (20,1), key="-CREATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    new_project_window = sg.Window("Create New Project", layout, modal=True)
    choice = None
    
    while True:
        event, values = new_project_window.read()

        if event == '-CREATE-PROJECT-':
            ## Create the folder
            parent_folder = values['-NEW-PROJECT-FOLDER-']
            project_name  = values['-NEW-PROJECT-NAME-']
            
            project_folder = os.path.join(parent_folder, project_name)
            
            dialog_box = new_project_window["-DIALOG-"]
            

            dialog_box = new_project_window["-DIALOG-"]
            
            if os.path.exists(project_folder):
                dialog_box.update(value=dialog_box.get()+'\n - The project folder already exists.')
                break
            else:
                os.mkdir(project_folder)
                dialog_box.update(value=dialog_box.get()+'\n - New project folder has been created.')
                
            extension = values['-IMAGE-EXTENSION-']
            images_folder = values['-NEW-IMAGES-FOLDER-']
            
            temp_path = os.path.join(images_folder,r"**")
            temp_path = os.path.join(temp_path,r"*."+extension)
    
            image_full_paths = glob.glob(temp_path, recursive=True)
            image_names = [os.path.split(path)[1] for path in image_full_paths]
            
            dialog_box.update(value=dialog_box.get()+'\n - '+str(len(image_names))+' images found.')
            
            df_files = pd.DataFrame({'file name':image_names,'full path':image_full_paths})
            df_files["image quality"] = "undefined"
            df_files["notes"] = "none"
            df_files["annotated"] = "No"
            
            df_files = df_files.drop_duplicates(subset='file name', keep="first")
            
            df_files.to_csv(os.path.join(project_folder, df_files_name))
            df_files.to_csv(os.path.join(project_folder, df_files_name), index=False)
            dialog_box.update(value=dialog_box.get()+'\n - Dataframe with file names created.')
            
            reference_image_path = values['-NEW-REF-IMAGE-']
            
            try:
                new_ref_image_path = os.path.join(project_folder, ref_image_name)
                shutil.copy(reference_image_path, new_ref_image_path)
                dialog_box.update(value=dialog_box.get()+'\n - Reference image copied in the project folder.')
            except:
                dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Error occurred while copying reference image file."')
            
            new_model_path = values['-NEW-MODEL-FILE-']
            df_model = pd.read_csv(new_model_path)
            df_model.to_csv(os.path.join(project_folder, df_model_name))
            df_model.to_csv(os.path.join(project_folder, df_model_name), index=False)
            dialog_box.update(value=dialog_box.get()+'\n - "Dataframe with model information copied in the project folder.')
            
            try:
                landmark_names = df_model['name'].values
                df_landmarks = df_files[['file name']].copy()
                for landmark in landmark_names:
                    df_landmarks[landmark] = np.nan
                
                df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
                df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name), index=False)
                dialog_box.update(value=dialog_box.get()+'\n - "Dataframe for landmarks coordinates created.')
            except:
                dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Problem in the creation of the landmarks dataframe.')
 
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    new_project_window.close()
    
    return


def create_registration_window(shared,df_landmarks,df_model,df_files, df_lines):
    """
    Function used to register selected images.
    It opens a new graphical window and collects from the user the info required
    for the creation of the registered images: 
        
        - the images location
        - a folder to save the registered images
        - a slider to define the output images definition
          
    Finally, it creates all the new images in the target folder.
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [[sg.Text('Where to save the registered images ', size=(35, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-REGISTERED-IMAGES-FOLDER-'),
               sg.FolderBrowse()],
              [sg.Spin([i for i in range(1,10000)], initial_value=1,size=4, key = 'Alpha'), sg.Text(': Alpha'),
               sg.Spin([i for i in range(1,10000)], initial_value=2,size=4, key = 'Smoothing'), sg.Text(': Smoothing'),
               sg.Spin([i for i in range(-10000,10000)], initial_value=-5,size=4, key = 'w_lines'), sg.Text(': Attraction to white lines'), ],
              [sg.Text('Image resolution (%)',size=(20,1)),
              sg.Slider(orientation ='horizontal', key='-REGISTRATION-RESOLUTION-', range=(1,100),default_value=100)],
              [sg.Button("Register the images ", size = (10,1), key="-REGISTRATION-SAVE-")],
              [sg.ProgressBar(max_value=100, size=(30,10), key= "-PROGRESS-")],
              [sg.Column(layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]],scrollable=True)]
              ]
    
    
    df_files = pd.read_csv( os.path.join(shared['proj_folder'], df_files_name))
    registration_window = sg.Window("Register the annotated images", layout, modal=True)
    choice = None
    dialog_box = registration_window["-DIALOG-"]
    
    while True:
        event, values = registration_window.read()
        if event == '-REGISTRATION-SAVE-':
            alpha = values['Alpha']
            smoothing = values['Smoothing']
            w_line = values['w_lines']
            # getting the path/directory of the images using a path of the csv file
            folder_dir = str(os.path.dirname(df_files["full path"][0]))
            
            try:
                _, _, files = next(os.walk(folder_dir))
                file_count = len(files)
                os.chdir(values['-REGISTERED-IMAGES-FOLDER-'])
                
            except: 
                dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Missing path input.')
            
            progress_bar=0
            # Getting reference landmarks + snake
            c_dst=[]
            img = shared['ref_image']
            landmarks_list = df_model["name"].values
            for landmark in shared['list_landmarks']:
                [x,y] = ast.literal_eval(df_model.loc[df_model["name"]==landmark, "target"].values[0])
                c_dst.append([x,y])
            # Getting snake landmarks
            LM = []
            N=[]
            for i in range(len(df_lines["Lmk1"])-6):
            # for i in range(1):
                LMref=[]
                lm1 = ast.literal_eval(df_model.loc[df_model["name"]==df_lines["Lmk1"][i], "target"].values[0])
                lm2 = ast.literal_eval(df_model.loc[df_model["name"]==df_lines["Lmk2"][i], "target"].values[0])

                snk = snake_contour(img,lm1[1],lm1[0],lm2[1],lm2[0],alpha,smoothing,w_line)
                if len(snk)>=10:
                      x = math.ceil(len(snk)*1/3)
                      snk = [snk[0]] + [snk[x]] + [snk[2*x]] + [snk[-1]]
                      snk = [snk[x].tolist() for x in range(len(snk))]
                      
                LMref.append(snk[1:-1])
                
                N.append(len(snk))
                c_dst.extend(LMref[0])
            shape = shared['ref_image'].size
            c_dst = c_dst/np.asarray(shape)
            

            
            
            # Checking for missing landmarks
            NanLandmarks = df_landmarks[df_landmarks.isnull().any(axis=1)]['file name'].values  
            
            # get images and their landmarks
            k=0
            for images in os.listdir(folder_dir):
                if str(images) in NanLandmarks : 
                    k+=1
                    progress_bar += 1
                    dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Landmark missing for ' + str(images) + ' it has been skipped')
                    continue
                else : 
                    
                    pass
                path = df_files.loc[df_files["file name"]==str(images),"full path"].astype('string').values[0]
                img = cv2.imread(str(path))
                
                
                # get image landmarks
                c_src=[]
                for LM in landmarks_list:
                    
                    LM_position = df_landmarks.loc[df_landmarks["file name"]==str(images), LM].values[0]
                    try :
                        c_src.append(ast.literal_eval(LM_position))
                    except :
                        continue
                    
                    
                # Get snake image landmarks

                LM = []
                for i in range(len(df_lines["Lmk1"])-6):
                    LM = []
                    df_landmarks.loc[df_landmarks["file name"] == df_files["file name"],"LV2-end"].values[0]
                    lm1 = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"] == df_files["file name"][k],df_lines["Lmk1"][i]].values[0])
                    lm2 = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"] == df_files["file name"][k],df_lines["Lmk2"][i]].values[0])
                    # alpha = (df_lines.loc[df_lines["Lmk1"][i]==df_model["name"], "alpha"])

                    snk = [snake_contour(img,lm1[1],lm1[0],lm2[1],lm2[0], alpha, smoothing, w_line, N=N[i])]

                    snk = [snk[x].tolist() for x in range(len(snk))]
                    
                    LM.extend(snk)
                    c_src.extend(LM[0][1:-1])
                shape = img.size
                c_src = c_src/np.asarray([img.shape[1],img.shape[0]])
                
                # Resize the image according to the slider value
                size = img.shape[0:2]*np.array([values['-REGISTRATION-RESOLUTION-']/100,values['-REGISTRATION-RESOLUTION-']/100])
                size = [int(x) for x in size]
                img = cv2.resize(img,(size[1],size[0]))
                
                
                # Apply tps 

                try : 
                    warped = TPSwarping(img, c_src, c_dst, img.shape[0:2])
                except : 
                    dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Problem in the warping of' + str(images))
                    continue
                
                k+=1
                
                cv2.imwrite(str(images) , warped)
                
                try:
                    dialog_box.update(value=dialog_box.get()+'\n - ' + str(images) + ' has been registered')
            
                except:
                    dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Problem in the registration of' + str(images))
                    
                # update the loading bar
                progress_bar+=1
                registration_window["-PROGRESS-"].update((progress_bar/file_count)*100)
            
            dialog_box.update(value=dialog_box.get()+'\n - All of the images have been registered')
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    registration_window.close()
    
    return

def CNN_create(window,X_train,y_train,X_test,y_test,shared,values):
    '''
    Create a CNN model using the create_CNN function

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    X_train : Array of iamge training data
    X_test : Array of image testing data
    y_train : Coordinates array of training data
    y_test : Coordinates array of testing data
    shared : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    window['-MODEL-RUN-STATE-'].update('Initializing...', text_color=('yellow'))
    window.Refresh()
    image_registration.create_CNN(X_train,y_train,X_test,y_test,shared['proj_folder'], values["-EPOCHS-"], shared['curr_image'].size, window, values)
    window['-MODEL-RUN-STATE-'].update('No', text_color=('red'))
    window.Refresh()
    
def CNN_continue(window,X_train,y_train,X_test,y_test,shared,values):
    '''
    Continue the training of a model using the continue_CNN function

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    X_train : Array of iamge training data
    X_test : Array of image testing data
    y_train : Coordinates array of training data
    y_test : Coordinates array of testing data
    shared : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    window['-MODEL-RUN-STATE-'].update('Initializing...', text_color=('yellow'))
    window.Refresh()
    image_registration.continue_CNN(X_train,y_train,X_test,y_test,values["-MODEL-FOLDER-"], values["-EPOCHS-"], window, values)
    window['-MODEL-RUN-STATE-'].update('No', text_color=('red'))
    window.Refresh()
    
def rotate(coord, angle, img_size):
    """
    Rotate a list of coordinates counterclockwise by a given angle in radians around the image center.

    Parameters :

        list = list of landmarks coordinates in an array
        angle = angle of rotation (clockwise)
        img_size = tuple of the image shape

    Returns :

        outlist = list of all the modified landmarks coordinates in an array
    """
    outlist=[]
    ox, oy = img_size[1]/2, img_size[0]/2
    for i in range(len(coord)):
        px, py = coord[i]
        qx = round(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy),2)
        qy = round(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy),2)
        outlist.append([qx,qy])
    return outlist

def data_augmentation(shared, df_landmarks, df_files, df_landmarks_name, augmented_by_x):
    
    # Getting the global project folder
    os.chdir(shared['proj_folder'])
    global_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(global_folder)
    
    # Creating the folder for the images + csv file
    try : 
        os.mkdir('augmented_data') 
    except : 
        print('folder already exist')
        print(global_folder)
        print(global_folder + '/augmented_data/*')
        files = glob.glob(global_folder + '/augmented_data/*')
        print(files)
        for f in files:
            os.remove(f)


    # Getting into the images folder

    os.chdir(os.path.normpath(df_files["full path"][0] + os.sep + os.pardir))


    # shutil.copy(shared['proj_folder'] + '/' + df_landmarks_name,global_folder + '/augmented_data/')

    df_landmarks_np = pd.read_csv(shared['proj_folder'] + '/' + df_landmarks_name).to_numpy()

    for i in range(len(df_landmarks_np)):
        for j in range(len(df_landmarks_np[i])-1):
            df_landmarks_np[i][j+1] = ast.literal_eval((df_landmarks_np[i][j+1]))
        
    output_landmarks= []
    for i in range(augmented_by_x * len(df_landmarks)):
        angle = rd.randint(0,359)
        img = Image.open(df_landmarks_np[i%len(df_landmarks)][0])
        rot = img.rotate(angle)
        rot.save(global_folder + '/augmented_data/' + str(i)+df_landmarks_np[i%len(df_landmarks)][0])
        clist = rotate(df_landmarks_np[i%len(df_landmarks)][1:], math.radians(-angle), (img.size[0],img.size[1]))
        clist.insert(0,(str(i) + df_landmarks_np[i%len(df_landmarks)][0]))
        output_landmarks.append(clist)

    rot_lm_df = pd.DataFrame(output_landmarks, columns = df_landmarks.columns.values)
    rot_lm_df.to_csv(global_folder + '/augmented_data/' + 'landmarks.csv', index=False)


def merge_projects():
    """
    Function used to merge two existing projects.
    It opens a new graphical window where the user sleect the paths to the two
    proojects to merge and the path where to create the new project.

    Finally, it creates all the new project files in the target folder.
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [[sg.Text("Project name: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-NEW-PROJECT-NAME-')],
              
              [sg.Text('Project location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-PROJECT-FOLDER-'),
               sg.FolderBrowse()],
              
              [sg.Text('Location of Project 1: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-PROJECT-FOLDER-1-'),
               sg.FolderBrowse()], 
              
              [sg.Text('Location of Project 2: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-PROJECT-FOLDER-2-'),
               sg.FolderBrowse()], 

              [sg.Button("Create the project: ", size = (20,1), key="-CREATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    merge_projects_window = sg.Window("Create New Project", layout, modal=True)
    choice = None
    

    
    while True:
        event, values = merge_projects_window.read()

        if event == '-CREATE-PROJECT-':
            
            ## Create the folder
            parent_folder = values['-NEW-PROJECT-FOLDER-']
            project_name  = values['-NEW-PROJECT-NAME-']
            
            project_folder = os.path.join(parent_folder, project_name)
            
            dialog_box = merge_projects_window["-DIALOG-"]
            
            if os.path.exists(project_folder):
                dialog_box.update(value=dialog_box.get()+'\n - The project folder already exists.')
                break
            else:
                os.mkdir(project_folder)
                dialog_box.update(value=dialog_box.get()+'\n - New project folder has been created.')
                
            folder_1 = values['-PROJECT-FOLDER-1-']
            
            try:
                df_files_1      = pd.read_csv( os.path.join(folder_1, df_files_name) )
                df_landmarks_1  = pd.read_csv( os.path.join(folder_1, df_landmarks_name) )
                df_model_1      = pd.read_csv( os.path.join(folder_1, df_model_name) )
                reference_image_path = os.path.join(folder_1, ref_image_name)
            except:
                dialog_box.update(value=dialog_box.get()+'\n - Problem opening project files of the the first project.')
                    
            folder_2 = values['-PROJECT-FOLDER-2-']
            
            try:
            
                df_files_2      = pd.read_csv( os.path.join(folder_2, df_files_name) )
                df_landmarks_2  = pd.read_csv( os.path.join(folder_2, df_landmarks_name) )
                df_model_2      = pd.read_csv( os.path.join(folder_2, df_model_name) )
            except:
                dialog_box.update(value=dialog_box.get()+'\n - Problem opening project files of the the second project.')
               
            if set(df_landmarks_1.columns) == set(df_landmarks_2.columns):
                
                df_files = pd.concat([df_files_1, df_files_2])
                df_files = df_files.drop_duplicates(subset='file name', keep="first")
                
                df_landmarks = pd.concat([df_landmarks_1, df_landmarks_2])
                df_landmarks = df_landmarks.drop_duplicates(subset='file name', keep="first")     

                try:
                    new_ref_image_path = os.path.join(project_folder, ref_image_name)
                    shutil.copy(reference_image_path, new_ref_image_path)
                    df_files.to_csv(os.path.join(project_folder, df_files_name))
                    df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
                    df_model_1.to_csv(os.path.join(project_folder, df_model_name))
                    dialog_box.update(value=dialog_box.get()+'\n - Merged files saved in the destination folder.')
                    
                except:
                    dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Error occurred while merging the files."')
                
            else:
                dialog_box.update(value=dialog_box.get()+'\n - The projects are based on different models. Can not merge.')
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    merge_projects_window.close()
    
    return

def add_new_images(shared, df_files, df_landmarks, df_model):
    """
    Function used to add new images to an existing project
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [             
              [sg.Text("New image files extension: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-IMAGE-EXTENSION-')],
              
              [sg.Text('New images location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-IMAGES-FOLDER-'),
               sg.FolderBrowse()], 
              
              [sg.Button("Update the project: ", size = (20,1), key="-UPDATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    add_images_window = sg.Window("Add new images", layout, modal=True)
    choice = None
    
    while True:
        event, values = add_images_window.read()

        if event == '-UPDATE-PROJECT-':

            project_folder = shared['proj_folder']
        
            extension = values['-IMAGE-EXTENSION-']
            images_folder = values['-NEW-IMAGES-FOLDER-']
            
            temp_path = os.path.join(images_folder,r"**")
            temp_path = os.path.join(temp_path,r"*."+extension)
    
            image_full_paths = glob.glob(temp_path, recursive=True)
            image_names = [os.path.split(path)[1] for path in image_full_paths]
            
            dialog_box = add_images_window["-DIALOG-"]
            dialog_box.update(value=dialog_box.get()+'\n - '+str(len(image_names))+' images found.')
            
            temp_df_files = pd.DataFrame({'file name':image_names,'full path':image_full_paths})
            temp_df_files["image quality"] = "undefined"
            temp_df_files["notes"] = "none"
            temp_df_files["annotated"] = "No"
            
            df_files = pd.concat([df_files, temp_df_files])
            df_files = df_files.drop_duplicates(subset='file name', keep="first")
            df_files = df_files.reset_index(drop=True)
            df_files.to_csv(os.path.join(project_folder, df_files_name))
            dialog_box.update(value=dialog_box.get()+'\n - Dataframe with file names updated.')

            landmark_names = df_model['name'].values
            temp_df_landmarks = df_files[['file name']].copy()
            for landmark in landmark_names:
                temp_df_landmarks[landmark] = np.nan
                
            df_landmarks = pd.concat([df_landmarks, temp_df_files])
            df_landmarks = df_landmarks.reset_index(drop=True)
            df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
            dialog_box.update(value=dialog_box.get()+'\n - "Dataframe for landmarks coordinates updated.')

        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    add_images_window.close()
    
    return df_files, df_landmarks

def select_image(shared, df_files):
    """
    Function used to jump to a specific imag ein the current project.
    It creates a pop-up window and allows to search among all the file names 
    included in the current project.
    It returns an updated 'shared' dictionary where the index of the current image 
    has been updated to point to the selected file name.
          
    """
    try:
        names = df_files["file name"].values
    except:
        names = []
    
    layout = [  [sg.Text('Search an image:')],
            [sg.Input(do_not_clear=True, size=(20,1),enable_events=True, key='_INPUT_')],
            [sg.Listbox(names, size=(20,10), enable_events=True, key='_LIST_')],
            [sg.Button('Exit')]]

    select_image_window = sg.Window('Search').Layout(layout)
    # Event Loop
    while True:
        event, values = select_image_window.Read()
        if event is None or event == 'Exit':                # always check for closed window
            break
        if values['_INPUT_'] != '':                         # if a keystroke entered in search field
            search = values['_INPUT_']
            new_values = [x for x in names if search in x] 
            select_image_window.Element('_LIST_').Update(new_values)
        else:
            select_image_window.Element('_LIST_').Update(names)
        if event == '_LIST_' and len(values['_LIST_']):    
            chosen_file = values['_LIST_'][0]
            sg.Popup('Selected ', chosen_file)
            index = 0
            try:
                index = int( df_files[df_files["file name"]==chosen_file].index[0] )
            except:
                pass
            shared['im_index'] = index
            
    select_image_window.Close()
    return shared

def make_main_window(size, graph_canvas_width):
    """
    Function used to create the main window, conatins the definition of the layout
    of the GUI.
    
    Parameters
    ----------
    size : (int, int)
        width and height of the main window.
        
    graph_canvas_width : int, optional
        width of the graph elements where images are visualized, in pixels. The default is 700px.


    Returns
    -------
    None.

    """
    
    # --------------------------------- Define Layout ---------------------------------

    selection_frame = [[sg.Text('Open existing project: ', size=(20, 1)), 
                        sg.Input(size=(20,1), enable_events=True, key='-PROJECT-FOLDER-'),
                        sg.FolderBrowse(size=(20,1)),
                        sg.Button("Load selected project", size=(20,1), key='-LOAD-PROJECT-')],
                       [sg.VPush()],
                       [sg.Button("Registration", size = (20,1), key="-REGISTRATION-"),
                        sg.Button("Create New project", size = (18,1), key="-NEW-PROJECT-", pad=((135,0),(0,0))),
                        sg.Button("Add images to project", size = (20,1), key="-NEW-IMAGES-"),
                        sg.Button("Merge projects", size = (20,1), key="-MERGE-PROJECTS-")]]
    
    CNN_frame = [[sg.Text('Data augmentation : ', size=(20, 1)),
                  sg.Button("Augment by 1 times", size=(20,1), key='-DATA-AUG-'),
                  sg.Spin([s for s in range(1,366)],initial_value=1, size=3, enable_events=True, key = "-DATA-NUM-")
                  ],
                 [sg.Text('Create a new CNN', size=(20, 1)),
                  sg.Input(size=(2,1), enable_events=True, key='-IMG-FOLDER-'),
                  sg.FolderBrowse("Images folder",size=(12,1), key='-IMG-FOLDER-'), 
                  sg.Button("Create", size=(12,1), key='-CNN-CREATE-')],
                 [sg.Text('Continue with a pre-trained CNN', size=(20, 1)),
                  sg.Input(size=(2,1), enable_events=True, key='-IMG-FOLDER2-'),
                  sg.FolderBrowse("Images folder",size=(12,1)),
                  sg.Input(size=(2,1), enable_events=True, key='-MODEL-FOLDER-'),
                  sg.FileBrowse("Model file",size=(12,1)), 
                  sg.Button("Continue", size=(20,1), key='-CNN-CONTINUE-')]]

    epochs_frame = [[sg.Text('Number of epochs : ', size=(20, 1)),
                     sg.Spin([s for s in range(1,1000000)],initial_value=1, size=5, enable_events=True, key = "-EPOCHS-")],
                    [sg.Checkbox('Infinte epochs', default=False, key = "-INF-EPOCHS-")],
                    [sg.Text('Epochs left : X', size=(17, 1), key = '-EPOCHS-COUNT-'),
                    sg.Text('Current precision : Y ', size=(28, 1), key = '-CURRENT-MAE-')],
                    [sg.Text('Currently running :', size=(15, 1)), 
                     sg.Text('No', text_color=('red'), size= (30,1), key = "-MODEL-RUN-STATE-")],
                    [sg.Button('Stop training',size = (15,1), key = '-STOP-TRAINING-')],
                    [sg.Text('')],
                    [sg.Input(size=(2,1), enable_events=True, key='-IMG-FOLDER3-'),
                     sg.FolderBrowse("Images folder",size=(12,1)),
                     sg.Input(size=(2,1), enable_events=True, key='-MODEL-FOLDER2-'),
                      sg.FileBrowse("Model file",size=(12,1))],
                    [sg.Button('Landmarks detection', key ='LM-DETECT')]
                    ]
    
    
    image_column = [[sg.Text("Image:", size=(10, 1)), 
                     sg.Text("", key="-CURRENT-IMAGE-", size=(35, 1)),
                     sg.Button("Select image", key="-SELECT-IMAGE-")],
                    [sg.Checkbox('Normalize the image preview', key="-NORMALIZATION-", default=True, enable_events=True),
                     sg.Text("Change Brightness:", size=(20, 1)), 
                     sg.Slider(range=(1, 100), key = "-BRIGHTNESS-", orientation='h', size=(15, 20), default_value=100, enable_events=True,  disable_number_display=True)],
                    [sg.Graph(  canvas_size=(graph_canvas_width, graph_canvas_width),
                                graph_bottom_left=(0, 0),
                                graph_top_right=(graph_canvas_width, graph_canvas_width),
                                key="-GRAPH-",
                                enable_events=True,
                                background_color='white',
                                drag_submits=False)]]
                
    
    annotation_column = [[sg.Button("Next"), sg.Button("Previous"), sg.Button("Next not annotated")],
                         [sg.Text("Image quality:")],
                         [sg.Combo(values=image_quality_choiches, size=(35,10), enable_events=True, key='-IMAGE-QUALITY-')],
                         [sg.Text("Additional notes: ", size=(20, 1))], 
                         [sg.Input(size=(15, 1), key="-IMAGE-NOTES-")],
                         [sg.Text("Image annotated: ", size=(20, 1))], 
                         [sg.Combo(values=["No", "Yes"], size=(35,10), enable_events=True, key='-IMAGE-ANNOTATED-')],
                         [sg.Text("Annotation completion [%]: ")],
                         [sg.ProgressBar(max_value=100, size=(30,10), key= "-PROGRESS-")],
                         [sg.Text("Landmarks model preview: ")],
                         [sg.Graph(canvas_size=(300, 300),
                                graph_bottom_left=(0, 0),
                                graph_top_right=(300, 300),
                                key="-LANDMARKS-PREVIEW-",
                                enable_events=False,
                                drag_submits=False,
                                background_color='white')],
                         [sg.Checkbox('Draw Line', default = False, font = 'Arial 18', key='-LINE-',  enable_events=True)],
                         ]
       
    annotation_frame = [[sg.Column(image_column), sg.Column(annotation_column)]]
    
    communication_window = [[sg.Text("", key="-PRINT-", size=(100, 10))]]
    
    layout = [[sg.Frame("Select project: ", layout = selection_frame),
               sg.Frame("Neural network", layout = CNN_frame)],
              [sg.Frame("Annotate images: ", layout = annotation_frame),
               sg.Frame("Model parameters", layout = epochs_frame, vertical_alignment = 'top')],
              [sg.Button("Save changes to the project", key="-SAVE-")],
              [sg.Frame("Messages: ", layout = communication_window)]
              ]
    
    
    # --------------------------------- Create Window ---------------------------------
    
    return sg.Window("Image Annotation Tool", layout, size=size, finalize=True, return_keyboard_events=True)


def make_landmarks_window(model_df, landmarks_df, current_filename, location = (1200,100), size = (300,900)):
    """
    Function used to create the extra window with buttons used to select the 
    various landmarks. The window is created only when a project is open and is 
    recreated every time the current image is changed.
    
    Parameters
    ----------
    model_df : dataframe
    
    landmarks_df : dataframe
    
    current_file_name : str
    name of the current image
    
    location : (int, int), optional
    Position of the window. The default is (1200,100).
    
    size : (int, int), optional
    Size of the window. The default is (300,900).

    Returns
    -------
    None.

    """
    landmarks_list = model_df["name"].values
    landmarks_buttons_colors = []
    
    for LM in landmarks_list:
        
        LM_position = landmarks_df.loc[landmarks_df["file name"]==current_filename, LM].values[0]
        
        if LM_position != LM_position: # check if LM_position is np.nan
            landmarks_buttons_colors.append("FloralWhite") 
        else:
            landmarks_buttons_colors.append("SteelBlue3")     
        
    layout = [[sg.Text('Select landmark: ', size=(20, 1))],
              *[[sg.Button(LM, size=(20,1), key = LM, button_color = ("black", landmarks_buttons_colors[i])),] for i, LM in enumerate(landmarks_list)],
              [sg.Button("Show all landmarks", size = (20,1), key = "-SHOW-ALL-")],
              [sg.Button("Delete current Landmark", size = (20,1), key = "-DELETE_LDMK-", button_color = ("black", "orange"))],
             ]
    
    return sg.Window("Landmark selection", layout, size=size, finalize=True, location = location)

def mouse_click_callback(event, window):
    """
    Function that can be bound to a click event in a window. It checks if the click
    happened inside a window element and, if not, rises a -WINDOW-CLICK- event.
    It is used to define the behaviour of the program when the user clicks on 
    the background of the window.
    
    Parameters
    ----------
    event : str
        event name.
    window : PySimpleGUI window
        window where the event took place.

    Returns
    -------
    None.

    """
    x, y = window.TKroot.winfo_pointerxy()
    try:
        widget = window.TKroot.winfo_containing(x, y)
    except:
        widget = None
    for element in window.element_list():
        element_widget = element.Widget
        # if the mouse click happened inside any element of the window, simply 
        # return nothing:
        if widget  == element_widget:
            return
    # otherwise, rise a "-WINDOW-CLICK-" event
    window.write_event_value('-WINDOW-CLICK-', (x, y))
    return