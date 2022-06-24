#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:14:04 2022

@author: titouan
"""

import numpy as np
import cv2
from PIL import Image

def TPS(img, reference, c_src, c_dst, dshape=(512,512)):
    """
    NAME
    	Registration.TPS

    ============================================================

    Warps the image using the tps module.

    Parameters
    ----------
    img : image input
    c_src : coordinates of the landmarks on the original image
    c_dst : coordinates of the landmarks on the reference image
    dshape : size of the output image, (512,512) by default

    Returns
    -------
    warped : Warped image

    """

    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC,borderMode=0)
    warped = warp_image_cv(img, c_src, c_dst, dshape) #generate the warped image
    warped = Image.fromarray(warped)
    return warped

