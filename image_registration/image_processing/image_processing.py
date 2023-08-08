#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:55:06 2023

@author: ceolin
"""

from functools import partial
from scipy import optimize, ndimage
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt

def rebin(img, binning):
    """
    Function used to rebin an image
    
    Parameters
    ----------
    arr : numpy array
        an image stored as a numpy array.
    binning : int
        binning factor.

    Returns
    -------
    resized : numpy array
        the resized image.

    """
    width = int(img.shape[0] / binning)
    height = int(img.shape[1] / binning)
    dim = (width, height)
    resized = resize(img, dim, preserve_range=True, anti_aliasing=True)
    return resized


def open_image_PIL(image_path, normalize=True):
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

def open_image_numpy(image_path, normalize=True):
    """
    Opens the image file, converts it into 8 bit, and optionally, normalizes it 
    
    Parameters
    ----------
    image_path : str
        path to the image.
    normalize : bool, optional.
    Returns
    -------
    a numpy array.
    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = convert_image_to8bit(image, normalize)
    return image


def bin_normalize_smooth(img, binning, smoothing=None):
    """
    Apply binning, normalization, and optional smoothing to an image.

    Parameters:
        img (numpy.ndarray): The input image as a 2D numpy array.
        binning (int): The binning factor for image reduction.
        smoothing (float or None): The standard deviation of the Gaussian filter for smoothing.
                                   If None, no smoothing is applied. Default is None.

    Returns:
        numpy.ndarray: Processed image after binning, normalization, and optional smoothing.

    The function performs the following steps:
    1. Apply binning to reduce the image size.
    2. Optionally apply Gaussian smoothing if a smoothing value is provided.
    3. Normalize the image to the range [0, 1].
    The processed image is returned.
    """

    # Apply binning to reduce image size
    img = rebin(img, binning)

    # Optionally apply Gaussian smoothing
    if smoothing:
        img = gaussian(img, smoothing, preserve_range=False)

    # Normalize the image to the range [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    return img

def convert_image_to8bit(image, normalize=False):
    """
    Helper function to convert an image in 8 bit format and, optionally, normalize it
    
    Parameters
    ----------
    image : numpy array
    normalize : bool, optional
        whether the image shoudl be normalized. The default is False.
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


def enhance_and_extract_edges(image, large_scale_smoothing, small_scale_smoothing, min_object_size, invert=True):
    """
    Enhances edges in an image and extracts the resulting edge map.

    Parameters:
        image (numpy.ndarray): The input image as a 2D numpy array.
        large_scale_smoothing (float): The standard deviation of the Gaussian filter for large scale smoothing.
        small_scale_smoothing (float): The standard deviation of the Gaussian filter for small scale smoothing.
        min_object_size (int): The minimum size of objects to retain after removing small objects.
        invert (bool): Whether to invert the final edge map. Default is True.

    Returns:
        numpy.ndarray: A 2D numpy array representing the enhanced edge map.

    The function enhances edges in the image by applying Gaussian filters with different scales,
    thresholding, removing small objects, and computing the distance transform of the resulting edges.
    The enhanced edge map is returned as a 2D numpy array.
    """
    
    # Validate parameter ranges
    if large_scale_smoothing <= small_scale_smoothing:
        raise ValueError("The 'large_scale_smoothing' parameter must be greater than 'small_scale_smoothing'.")
    if min_object_size <= 0:
        raise ValueError("The 'min_object_size' parameter must be a positive integer.")

    # Compute the edges using Gaussian filtering
    edges = gaussian(image, large_scale_smoothing, preserve_range=True) - gaussian(image, small_scale_smoothing, preserve_range=True)

    # Threshold the edges
    threshold_value = 0.5 * threshold_otsu(edges)
    edges = edges > threshold_value

    # Remove small objects
    edges = remove_small_objects(edges, min_size=min_object_size)

    # Apply Gaussian filtering again
    edges = gaussian(edges, (large_scale_smoothing + small_scale_smoothing) / 2, preserve_range=False)

    # Threshold the edges again
    edges = edges > threshold_otsu(edges)

    # Compute the distance transform of the edges
    edges = distance_transform_edt(edges)
    edges = edges / np.max(edges)

    # Invert the edge map if requested
    if invert:
        edges = 1 - edges

    return edges

def active_contour_energy_fixed_boundaries(flattened_pts, p_start, p_end, energy_image, alpha):
    """
    Calculate the total energy of the active contour defined by a set of flattened points
    with fixed start and end points (boundaries).

    Parameters:
        flattened_pts (numpy.ndarray): A 1D array representing a list of flattened points (alternating y and x coordinates).
        p_start (tuple): A tuple of (y_start, x_start) representing the coordinates of the start point (boundary).
        p_end (tuple): A tuple of (y_end, x_end) representing the coordinates of the end point (boundary).
        energy_image (numpy.ndarray): The energy map as a 2D numpy array.
        alpha (float): The weight parameter for balancing the energies.

    Returns:
        float: The total energy of the active contour.

    The function reconstructs the active contour from the flattened points and adds the fixed start and end points.
    It calculates the spacing energy, curvature energy, length energy, and external energy for the active contour.
    The total energy is calculated as a weighted sum of these energies using the alpha parameter.
    """
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 2), 2))
    pts = np.vstack([p_start, pts, p_end])

    displacements = np.diff(pts, axis=0)
    point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    mean_dist = np.mean(point_distances)

    memo = {'displacements': displacements, 'point_distances': point_distances, 'mean_dist': mean_dist}

    # Spacing energy (favors equi-distant points)
    spacing_energy = average_spacing_energy(pts, memo)

    # Curvature energy (favors smooth curves)
    curvature_energy = average_curvature_energy(pts, memo) * (mean_dist ** 2)

    # Length energy:
    length_energy = mean_dist ** 2

    # External energy
    # Calculate the energy only at contour nodes:
    external_energy = average_energy_along_curve(pts, energy_image) * (mean_dist ** 2)

    total_energy = external_energy + alpha * (spacing_energy + length_energy + curvature_energy)

    return total_energy


def average_curvature_energy(pts, memo=None):
    """
    Calculate the average curvature energy of a curve defined by a set of points.

    Parameters:
        pts (numpy.ndarray): A 2D array of shape (n, 2) representing a list of points.
                             The y-coordinates are in the first column and x-coordinates are in the second column.
        memo (dict, optional): A dictionary containing cached computations for optimization. Default is None.

    Returns:
        float: The average curvature energy of the curve.

    The function calculates the average curvature energy of a curve defined by the points.
    If the memo is provided and contains precomputed values, it will use those values for optimization.
    """
    if isinstance(memo, dict):
        try:
            displacements = memo['displacements']
            point_distances = memo['point_distances']
        except KeyError:
            displacements = np.diff(pts, axis=0)
            point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    else:
        displacements = np.diff(pts, axis=0)
        point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    norm_displacements = displacements
    norm_displacements[:, 0] = norm_displacements[:, 0] / point_distances
    norm_displacements[:, 1] = norm_displacements[:, 1] / point_distances
    curvature_1d = np.diff(norm_displacements, axis=0)
    curvature = curvature_1d[:, 0] ** 2 + curvature_1d[:, 1] ** 2
    curvature_energy = (np.mean(curvature))
    return curvature_energy


def average_spacing_energy(pts, memo=None):
    """
    Calculate the average spacing energy of a curve defined by a set of points.

    Parameters:
        pts (numpy.ndarray): A 2D array of shape (n, 2) representing a list of points.
                             The y-coordinates are in the first column and x-coordinates are in the second column.
        memo (dict, optional): A dictionary containing cached computations for optimization. Default is None.

    Returns:
        float: The average spacing energy of the curve.

    The function calculates the average spacing energy of a curve defined by the points.
    If the memo is provided and contains precomputed values, it will use those values for optimization.
    """
    if isinstance(memo, dict):
        try:
            displacements = memo['displacements']
            point_distances = memo['point_distances']
            mean_dist = memo['mean_dist']
        except KeyError:
            displacements = np.diff(pts, axis=0)
            point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
            mean_dist = np.mean(point_distances)
    else:
        displacements = np.diff(pts, axis=0)
        point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
        mean_dist = np.mean(point_distances)

    spacing_energy = np.mean((point_distances - mean_dist) ** 2)
    return spacing_energy


def average_energy_along_curve(pts, energy_map):
    """
    Calculate the average energy of an energy map along a curve defined by a set of points.

    Parameters:
        pts (numpy.ndarray): A 2D array of shape (n, 2) representing a list of points.
                             The y-coordinates are in the first column and x-coordinates are in the second column.
        energy_map (numpy.ndarray): The energy map as a 2D numpy array.

    Returns:
        float: The average energy of the energy map along the curve.

    The function calculates the average energy of an energy map along a curve defined by the points.
    It uses interpolation to sample energy values from the energy map along the curve.
    """
    energy_values = ndimage.interpolation.map_coordinates(energy_map, [pts[:, 0], pts[:, 1]], order=1)
    return np.mean(energy_values)


def average_energy_along_curve_interp(flattened_pts, energy_map, memo=None):
    """
    Calculate the average energy of an energy map along a curve defined by a set of flattened points.

    Parameters:
        flattened_pts (numpy.ndarray): A 1D array representing a list of flattened points (alternating y and x coordinates).
        energy_map (numpy.ndarray): The energy map as a 2D numpy array.
        memo (dict, optional): A dictionary containing cached computations for optimization. Default is None.

    Returns:
        float: The average energy of the energy map along the curve defined by the flattened points.

    The function calculates the average energy of an energy map along a curve defined by the flattened points.
    It uses interpolation to reconstruct the curve from the flattened points and then samples energy values
    from the energy map along the curve.
    """
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 2), 2))
    
    if isinstance(memo, dict):
        try:
            displacements = memo['displacements']
            point_distances = memo['point_distances']
        except KeyError:
            displacements = np.diff(pts, axis=0)
            point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    else:
        displacements = np.diff(pts, axis=0)
        point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    cumul_dist = np.cumsum(point_distances)
    cumul_dist = np.array([0, *list(cumul_dist)])
    new_x = np.interp(np.arange(int(cumul_dist[-1])), cumul_dist, pts[:, 0])
    new_y = np.interp(np.arange(int(cumul_dist[-1])), cumul_dist, pts[:, 1])
    energy_values = ndimage.interpolation.map_coordinates(energy_map, [new_x, new_y], order=1)
    return np.mean(energy_values)



def crop_image_around_active_contour(image, pts, margin=0.10):
    """
    Crop an image around the active contour defined by the points (pts).

    Parameters:
        image (numpy.ndarray): The input image as a 2D numpy array.
        pts (numpy.ndarray): A 2D array of shape (n, 2) representing the points of the active contour.
                             The y-coordinates are in the first column and x-coordinates are in the second column.
        margin (float): The margin (in percentage) to expand the bounding box around the active contour.
                        It must be between 0 and 0.5. Default is 0.10.

    Returns:
        tuple: A tuple containing the cropped image as a 2D numpy array and the [x, y] coordinates of the top-left corner
               of the cropped region in the original image.

    The function calculates the bounding box around the active contour points, expands it by the specified margin,
    and crops the corresponding region from the input image. The cropped image and the [x, y] coordinates of the
    top-left corner of the cropped region are returned as a tuple.
    """

    # Validate the margin to be between 0 and 0.5
    margin = min(max(0, margin), 0.5)

    # Calculate the bounding box around the active contour with the specified margin
    xmin = int((1 - margin) * np.min(pts[:, 1]))
    xmax = int((1 + margin) * np.max(pts[:, 1]))
    ymin = int((1 - margin) * np.min(pts[:, 0]))
    ymax = int((1 + margin) * np.max(pts[:, 0]))

    # Get the dimensions of the input image
    h, w = image.shape

    # Ensure the bounding box is within the image boundaries
    h_max = min(ymax, h)
    h_min = max(ymin, 0)
    w_max = min(xmax, w)
    w_min = max(xmin, 0)

    # Crop the image based on the bounding box
    image_cropped = image[h_min:h_max, w_min:w_max]

    return image_cropped, [w_min, h_min]


def position_starting_points_active_contour(pts_ref, p_start, p_end, flip_pts = False):
    """
    Applies an affine transformation on the given set of points (pts_ref) to align its first and last points
    perfectly with the specified start and end points (p_start and p_end).

    Parameters:
        pts_ref (numpy.ndarray): A 2D array of shape (n, 2) representing a list of points.
                                 The y-coordinates are in the first column and x-coordinates are in the second column.
        p_start (tuple): A tuple of (p_start_y, p_start_x) representing the coordinates of the desired starting point.
        p_end (tuple): A tuple of (p_end_y, p_end_x) representing the coordinates of the desired ending point.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 2) representing the transformed points after applying the affine transformation.

    The function performs the following steps:
    1. Translates the points to make the first point of pts_ref coincide with p_start.
    2. Computes the scaling factor to achieve uniform scaling on both axes.
    3. Calculates the rotation angle required to align the last point of pts_ref with p_end.
    4. Builds the rotation matrix based on the rotation angle.
    5. Applies the affine transformation to the translated points using the rotation matrix and scaling factor.
    """
    
    if flip_pts:
        pts_ref[:, 1] = -pts_ref[:, 1]
    
    # Extract the y and x coordinates from pts_ref
    y_ref, x_ref = pts_ref[:, 0], pts_ref[:, 1]

    # Translate the points to make the first point of pts_ref coincide with p_start
    pts_ref -= [y_ref[0], x_ref[0]]

    # Compute the scaling factor (uniform scaling on both axes)
    scaling_factor = np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]) / np.hypot(y_ref[-1] - y_ref[0], x_ref[-1] - x_ref[0])
    pts_ref = scaling_factor * pts_ref

    # Compute the rotation angle
    angle_start_end = np.arctan2(p_end[0] - p_start[0], p_end[1] - p_start[1])
    angle_ref = np.arctan2(pts_ref[-1, 0], pts_ref[-1, 1])
    rotation_angle = angle_start_end - angle_ref

    # Build the affine transformation matrix with scaling and rotation
    cos_theta, sin_theta = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    # Apply the affine transformation to pts_ref
    transformed_pts_ref = np.dot(pts_ref, rotation_matrix.T) + np.array(p_start)

    return transformed_pts_ref

def fit_active_contour(energy, p_start, p_end, pts_relative_prior, flip_pts, 
                       alpha, spacing, 
                       options={'maxiter': 200, 'gtol':0.01, 'eps':0.01},
                       method='CG'):
    """
    Fit an active contour to an energy image using optimization.

    Parameters:
        energy (numpy.ndarray): The energy image as a 2D numpy array.
        p_start (tuple): Starting point coordinates (p_start_y, p_start_x).
        p_end (tuple): Ending point coordinates (p_end_y, p_end_x).
        pts_relative_prior (numpy.ndarray): Relative positions of prior points as a 2D numpy array.
        flip_pts (bool): Whether to flip the prior points horizontally.
        alpha (float): Weight parameter for the energy term in the optimization.
        spacing (float): Desired spacing between points along the contour.
        options (dict): Optimization options (default values provided).
        method (str): Optimization method to use (default is 'CG').

    Returns:
        numpy.ndarray: Optimized points representing the fitted active contour.

    The function performs the following steps:
    1. Define the initial positions for the active contour using relative prior points and spacing.
    2. Crop the energy image around the contour and normalize it.
    3. Find the contour with minimal energy using optimization.
    """

    # 1. Define the active contour initial positions:
    pts = position_starting_points_active_contour(pts_relative_prior, p_start, p_end, flip_pts=flip_pts)
    displacements = pts[1:] - pts[0:-1]
    point_distances = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    cumul_dist = np.cumsum(point_distances)
    cumul_dist = np.insert(cumul_dist, 0, 0)
    tot_length = np.sum(point_distances)
    n_points = int(tot_length / spacing)
    new_x = np.interp(np.arange(int(n_points + 1)) * (cumul_dist[-1]) / n_points, cumul_dist, pts[:, 0])
    new_y = np.interp(np.arange(int(n_points + 1)) * (cumul_dist[-1]) / n_points, cumul_dist, pts[:, 1])
    pts = np.array([new_x, new_y]).T
    
    # 2. Crop the image around the contour:
    energy_cropped, [x0, y0] = crop_image_around_active_contour(energy, pts, margin=0.05)
    energy_cropped = energy_cropped - np.min(energy_cropped)
    energy_cropped = energy_cropped / np.max(energy_cropped)
    pts_cropped = pts - [y0, x0]
    p_start = pts_cropped[0]
    p_end = pts_cropped[-1]
    
    # 3. Find the contour with minimal energy:
    cost_function = partial(active_contour_energy_fixed_boundaries, p_start=p_start,
                            p_end=p_end, alpha=alpha, energy_image=energy_cropped)
    
    res = optimize.minimize(cost_function, pts_cropped[1:-1].ravel(), method=method, options=options)
    optimal_pts = np.reshape(res.x, (int(len(res.x) / 2), 2))
    pts = np.vstack([p_start, optimal_pts, p_end])
    pts = pts + [y0, x0]
    
    return pts