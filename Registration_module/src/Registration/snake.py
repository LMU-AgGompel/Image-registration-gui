import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

def snake(img, landmark, sliding_lm_nb):
    """
    NAME
    	Registration.snake

    ============================================================

    Function using active_contour to detect dark edges in a image, and returning an array of intermediate landmarks between to point on the detected edge

    Parameters
    ----------
    img : input image
    landmark : list of the starting and ending point landmarks (ex : [[681,215],[685,930]] )
    sliding_lm_nb : number of sliding landmarks wanted

    Returns
    -------
    LM : list of the intermediate landmarks coordinates

    """
    
    img = rgb2gray(img)
    #Define the starting and ending points, as well as the number of interpolated landmarks
    r = np.linspace(landmark[0][0], landmark[1][0], sliding_lm_nb+2 )
    c = np.linspace(landmark[0][1], landmark[1][1], sliding_lm_nb+2 )
    
    init = np.array([r, c]).T

    #Create the snake 
    snake = active_contour(gaussian(img, 7, preserve_range=False),
                       init, boundary_condition='fixed',
                       alpha=20, beta=50.0, w_line=-5, w_edge=10, gamma=0.1, max_px_move=40)

    #Return the landmarks on the detected edge
    LM = []
    for i in range(sliding_lm_nb):
        LM.append([snake[i+1][1],snake[i+1][0]])
    return LM
