'''
  A graphical interface to annotate, place landmarks and register images, 
  built with the PySimpleGUI TKinter framework.
  
  Created on Fri Mar 25 18:13:00 2022
  
  @author: Stefano
'''

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import image_registration
from ._gui_helpers import *


def start_image_registration_GUI(main_window_size = (1200,1100), graph_canvas_width = 700):
    """
    Parameters
    ----------
    main_window_size : (int, int), optional
        width and height of the main window in pixels. The default is (1200,1100).
    graph_canvas_width : int, optional
        width of the graph elements where images are visualized, in pixels. The default is 700px.

    Initialize the image registration graphical interface and runs the main loop.
    
    Returns
    -------
    None.

    """

    # initialize the main window:
    main_window = make_main_window(main_window_size, graph_canvas_width)
    landmarks_window = None
    
    # The following command binds a click on any position in the main window to
    # a call to the "callback" function, which rises a -WINDOW-CLICK- event if 
    # the click happened outside main window elements:
    main_window.TKroot.bind("<ButtonRelease-1>", lambda x: mouse_click_callback(x, main_window), add='+')
    
    # Define variables shared between windows and their initial values:
    shared = {'im_index':0, 'curr_file': None, 'proj_folder': "", 'ref_image': None, 
              'curr_image': None, 'raw_image': None, 'curr_landmark': None, 'prev_landmark': None, 
              'list_landmarks': None, 'drawing_line': False, 'pt_size': int(graph_canvas_width/15), 
              'ref_img_pt_size': 30, 'normalize': True, 'edge_det_sigma_s': 10,  
              'edge_det_sigma_l': 50, 'edge_det_min_size': 1000, 'lmk_fine_tuning_max_dist': 30,
              'graph_width': graph_canvas_width, 'CNN_binning':10,
              'CNN_augmentation': 16, 'CNN_model': None, 'ref_floating_lmks':None}

    df_files = None
    df_landmarks = None
    df_model = None
    df_predicted_landmarks = None
    
    # Variable used to store previous event, used to allow for keyboard inputs
    # with simultaneous keys, like Ctrl+S, etc..
    previous_event = ""
    event = ""

    # --------------------------------- Event Loop ---------------------------------
    
    while True:
        
        previous_event = event
        window, event, values = sg.read_all_windows()

        if event == sg.WIN_CLOSED or event == 'Exit':
            # if closing the main window, exit program
            if window == main_window:    
                break
            # otherwise, do nothing
            else:                     
                pass
        
        if event == '-LOAD-PROJECT-':
            
            shared['proj_folder'] = values['-PROJECT-FOLDER-']

            # try to open all the project files and initialize variables:
            try:
            
                df_files       = pd.read_csv( os.path.join(shared['proj_folder'], df_files_name) )
                df_landmarks   = pd.read_csv( os.path.join(shared['proj_folder'], df_landmarks_name) )
                df_model       = pd.read_csv( os.path.join(shared['proj_folder'], df_model_name) )
                
                if os.path.exists( os.path.join(shared['proj_folder'], df_predicted_landmarks_name) ) :
                    df_predicted_landmarks = pd.read_csv(os.path.join(shared['proj_folder'], df_predicted_landmarks_name))
                
                shared['ref_image'] = open_image_PIL(os.path.join(shared['proj_folder'], ref_image_name), normalize=False)
                shared['im_index'] = 0
                
                shared['list_landmarks'] = df_model["name"].values
                
                # I need to convert columns of landmarks into object columns to be able to edit them later on:
                for landmark in shared['list_landmarks']:
                    df_landmarks[landmark] = df_landmarks[landmark].astype(object)
                
                # update graph objects, landmark window, and all fields related to the image:
                shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)               

                
            except Exception as error_message:
                print(error_message)
                main_window["-PRINT-"].update("An error occured while opening the new project!")
                pass
                
            
        if event == '-NEW-PROJECT-':
            create_new_project()
            
        if event == '-NEW-IMAGES-':
            if  (df_files is not None) and (shared['im_index']>0):
                df_files, df_landmarks = add_new_images(shared, df_files, df_landmarks, df_model)
                shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)

        if event == '-MERGE-PROJECTS-':
            merge_projects()
            
        if event == '-REGISTRATION-':
            registration_window(shared, df_landmarks, df_predicted_landmarks, df_model, df_files)
        
        if event == '-SELECT-IMAGE-':
            if (df_files is not None):
                shared = select_image(shared, df_files)
                shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)
            
        if event == "Next":
            if (df_files is not None) and (shared['im_index'] < (len(df_files.index)-1)):
                shared['im_index'] += 1
                shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)
                
        if event == "Previous":
            if  (df_files is not None) and (shared['im_index']>0):
                shared['im_index'] -= 1
                shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)
        
        if event == "Next not annotated":
            if df_files is not None:
                
                df_files = df_files.sort_values(by=['annotated', "file name"], ascending=False)
                df_files = df_files.reset_index(drop=True)
                indeces  = df_files[df_files['annotated']=="No"].index
                
                if len(indeces) > 0:
                    shared['im_index'] = indeces[0]
                    shared, landmarks_window = refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, df_predicted_landmarks, main_window, landmarks_window)
        
        if event == "-NORMALIZATION-":
            shared['normalize'] = values['-NORMALIZATION-']
            
        if event == "-IMAGE-QUALITY-":
            quality = values['-IMAGE-QUALITY-']
            df_files.loc[shared['im_index'], "image quality"] = quality           
        
        if event == "-IMAGE-NOTES-":
            annotation = values['-IMAGE-NOTES-']
            df_files.loc[shared['im_index'], "notes"] = annotation
            
        if event == "-IMAGE-ANNOTATED-":
            annotated = values['-IMAGE-ANNOTATED-']
            df_files.loc[shared['im_index'], "annotated"] = annotated
            
        if event == "-BRIGHTNESS-":
            if shared['raw_image']:
                scaling = 101/(1+values['-BRIGHTNESS-'])
                shared['curr_image'] = np.uint8(np.asarray(shared['raw_image'])*scaling)
                shared['curr_image'] = PIL.Image.fromarray(shared['curr_image'])
                update_image_view(shared['curr_image'], main_window, graph_canvas_width)

        if event == "-SAVE-" or ("Control" in previous_event and "s" in event):
            # Ctr-s keyboard shortcut or clicking to save button save the current
            # project.
            try:
                database_path = os.path.join(shared['proj_folder'], df_files_name)
                df_files.to_csv(database_path, index= False)
                database_path = os.path.join(shared['proj_folder'], df_landmarks_name)
                df_landmarks.to_csv(database_path, index= False)
                
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                main_window["-PRINT-"].update("** Project saved at "+current_time+" **")
            except:
                pass

        # --------------------- events related to the CNN ---------------------

        if event == '-CNN-CREATE-':
            binning =  values['-CNN-BIN-']
            image_shape = image_registration.check_image_shape(df_files)
            if image_shape:
                model_input_shape = [int(image_shape[0]/binning), int(image_shape[1]/binning)]
                shared['CNN_model'] = CNN_create(main_window, model_input_shape, df_model)

        if event == '-CNN-PATH-':
            CNN_load(main_window, values, shared)

        if event == '-CNN-TRAIN-':
            
            train_folder = os.path.join(shared['proj_folder'], "training_data")
            val_folder   = os.path.join(shared['proj_folder'], "validation_data")
            
            n_augment = values['-CNN-AUGM-']
            binning =  values['-CNN-BIN-']
            
            image_registration.training_data_preprocessing(train_folder, val_folder, df_landmarks, df_files, df_model, n_augment, binning, test_size=0.2, normalization=True)
            CNN_train(main_window, train_folder, val_folder, df_model, shared, values)

        if event == '-CNN-CONTINUE-TRAIN-':
            
            train_folder = os.path.join(shared['proj_folder'], "training_data")
            val_folder   = os.path.join(shared['proj_folder'], "validation_data")
            
            if os.path.isdir(train_folder) & os.path.isdir(val_folder):
                if (not os.listdir(train_folder)) & (not os.listdir(val_folder)):
                    CNN_train(main_window, train_folder, val_folder, df_model, shared, values)
            else:
                main_window["-PRINT-"].update("** No training and validation data to continue the training. \n Consider training from scratch**")
            
        if event == '-CNN-FINE-TUNE-':
            
            train_folder = os.path.join(shared['proj_folder'], "training_data")
            val_folder   = os.path.join(shared['proj_folder'], "validation_data")
            
            n_augment = values['-CNN-AUGM-']
            binning =  values['-CNN-BIN-']
            
            image_registration.training_data_preprocessing(train_folder, val_folder, df_landmarks, df_files, df_model, n_augment, binning, test_size=0.2, normalization=True)
            CNN_fine_tune(main_window, train_folder, val_folder, df_model, shared, values)
           


        if event == 'LM-DETECT':
            CNN_predict_landmarks(df_files, df_model, main_window, shared, values)
            df_predicted_landmarks = pd.read_csv(os.path.join(shared['proj_folder'], df_predicted_landmarks_name))
                
        if event == 'LM-FINETUNE':
            lmk_fine_tuning_window(shared, df_landmarks, df_predicted_landmarks, df_model, df_files)
            
        if event == 'LM-FLOATING':
            df_contours_model = pd.read_csv( os.path.join(shared['proj_folder'], df_contour_model_name ) )
            shared = floating_lmks_detection(shared, df_model, df_contours_model, df_files, df_landmarks)
            df_floating_landmarks = pd.read_csv(os.path.join(shared['proj_folder'], df_floating_landmarks_name), index= False)
               

        # -------------------- keyboard shortcuts: ----------------------------

        if ("Control" in previous_event and "a" in event):
            # Ctrl-a keyboard shortcut to set the "image annotated" field to "Yes"
            main_window["-IMAGE-ANNOTATED-"].update(value="Yes")
            try:
                df_files.loc[shared['im_index'], "annotated"] = "Yes"
            except:
                pass
        
        if ("Control" in previous_event and "g" in event):
            # Ctrl-g keyboard shortcut to set the image quality to "good"
            quality = "good"
            main_window['-IMAGE-QUALITY-'].update(value=quality)
            try:
                df_files.loc[shared['im_index'], "image quality"] = quality
            except:
                pass
            
        if ("Control" in previous_event and "f" in event):
            # Ctrl-f keyboard shortcut to set the image quality to "fair"
            quality = "fair"
            main_window['-IMAGE-QUALITY-'].update(value=quality)
            try:
                df_files.loc[shared['im_index'], "image quality"] = quality
            except:
                pass
            
        if ("Control" in previous_event and "p" in event):
            # Ctrl-p keyboard shortcut to set the image quality to "poor"
            quality = "poor"
            main_window['-IMAGE-QUALITY-'].update(value=quality)
            try:
                df_files.loc[shared['im_index'], "image quality"] = quality
            except:
                pass
            
        if ("Control" in previous_event and "b" in event):
            # Ctrl-g keyboard shortcut to set the image quality to "bad"
            quality = "bad"
            main_window['-IMAGE-QUALITY-'].update(value=quality)
            try:
                df_files.loc[shared['im_index'], "image quality"] = quality
            except:
                pass
            
        if ("Control" in previous_event and "2" in event):
            # Ctrl-l keyboard shortcut to loop through the landmarks
            try:
                if shared['curr_landmark']:
                    
                    landmark_index = np.where(shared['list_landmarks'] == shared['curr_landmark'])[0]
                    landmark_index += 1
                    
                    if landmark_index >= len(shared['list_landmarks']):
                        landmark_index = [0]
                    
                    # updating the event variable will trigger the same action as if the
                    # corresponding landmark button has been pressed
                    event = shared['list_landmarks'][landmark_index][0]
                
                else:
                    landmark_index = [0]
                    event = shared['list_landmarks'][landmark_index][0]
            except:
                pass
            
    # --------- interact with the graph to draw landmark points ---------------
    
        if event == "-GRAPH-":  
        # A graph event corresponds to a mouse click in the graph area
            
            x, y = values["-GRAPH-"]
            
            # refresh the graph to remove previous points:
            if shared['curr_image']:
                update_image_view(shared['curr_image'], main_window, graph_canvas_width)
            
                if values['-LINE-']:
                    if not shared['drawing_line']:
                        (x1,y1) = (x, y)
                        shared['drawing_line'] = True
                    else:
                        (x2, y2) = (x, y)
                        if x2==x1:
                            x2+=1
                        m = (y2-y1)/(x2-x1)
                        q = y1-m*x1
                        if m<0:
                            main_window['-GRAPH-'].draw_line((0,q), (-q/m, 0), width=4)
                        elif m>0:
                            main_window['-GRAPH-'].draw_line((shared['curr_image'].width,m*shared['curr_image'].width+q), (-q/m, 0), width=4)
                        else:
                            main_window['-GRAPH-'].draw_line((shared['curr_image'].width,m*shared['curr_image'].width+q), (0, q), width=4)
                        
                        shared['drawing_line'] = False
                    
                else:
                    # draw a dot on the graph
                    main_window['-GRAPH-'].draw_point((x,y), size = shared['pt_size'], color = "red")
                    
                    # if a landmark was selected in the landmark window we store the new position in the
                    # corresponding dataframe:
                        
                    if shared['curr_landmark']:
                        [x,y] = convert_graph_coordinates_to_image(x, y, shared['curr_image'].width, shared['curr_image'].height)
                        main_window["-PRINT-"].update('Position of landmark '+shared['curr_landmark']+' set to: ' + str([x, y]))
                        df_landmarks.loc[df_landmarks["file name"]==shared['curr_file'], shared['curr_landmark']] = str([x,y])
                        

        
        # ------ events related to elements of the landmarks window -----------
        
        try:   
            if event in shared['list_landmarks']:
                # when clicking on one of the buttons corresponding to the single
                # landmarks, we update the current landmark and show it as a dot on the
                # reference image.
                shared['curr_landmark'] = event

                # update colors of landmark buttons:
                landmarks_window[shared['curr_landmark']].update(button_color = ("black", "red"))

                if shared['prev_landmark']:
                    
                    LM_position = df_landmarks.loc[df_landmarks["file name"]==shared['curr_file'],  shared['prev_landmark']].values[0]
                    
                    if LM_position != LM_position: # check if LM_position is np.nan
                        landmarks_window[shared['prev_landmark']].update(button_color = ("black", "FloralWhite"))    
                    else:
                        landmarks_window[shared['prev_landmark']].update(button_color = ("black", "SteelBlue3"))
                                                                   
                shared['prev_landmark'] = shared['curr_landmark']
                update_landmarks_preview(shared, main_window, 300)
                
                draw_landmark_preview(main_window, df_model, shared, color = "red", size = shared['ref_img_pt_size'])
                
                update_image_view(shared['curr_image'], main_window, graph_canvas_width)

                try:
                    draw_landmark(main_window, df_landmarks, shared, color = "blue", size = shared['pt_size'])
                    
                    if df_predicted_landmarks is not None:
                        draw_landmarks_all(main_window, df_predicted_landmarks, shared, color = "green", size = shared['pt_size'])
                        
                except:
                    pass

                main_window["-PRINT-"].update('You are currently editing the position of landmark '+shared['curr_landmark'])
                
        except:
             pass
         
            
        if event == "-SHOW-ALL-":
            if shared['curr_image']:
                update_image_view(shared['curr_image'], main_window, graph_canvas_width)
                
            draw_landmarks_preview_all(main_window, df_model, shared, color = "red", size =  shared['ref_img_pt_size'])
            draw_landmarks_all(main_window, df_landmarks, shared, color = "blue", size = shared['pt_size'])
            
            if df_predicted_landmarks is not None:
                draw_landmarks_all(main_window, df_predicted_landmarks, shared, color = "green", size = shared['pt_size'])
                
        if event == "-DELETE_LDMK-":
            if shared['curr_landmark']:                
                df_landmarks.loc[df_landmarks["file name"]==shared['curr_file'],  shared['curr_landmark']] = np.nan
        
            
        if event == "-WINDOW-CLICK-":
        # remove the focus from any window element that was previously selected:
            try:
                x = window.FindElementWithFocus()
                x.block_focus()
            except:
                pass


    try:
        main_window.close()
        landmarks_window.close()
        
    except:
        pass
    return
