#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:07:30 2024

@author: ceolin
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.math cimport log

cpdef np.ndarray[double, ndim=2] cython_d(np.ndarray[double, ndim=2] a, np.ndarray[double, ndim=2] b):
    cdef np.ndarray[double, ndim=2] squared_distances = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    cdef int i, j

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            squared_distances[i, j] = (a[i, 0] - b[j, 0]) ** 2 + (a[i, 1] - b[j, 1]) ** 2

    return np.sqrt(squared_distances)


cpdef np.ndarray[double, ndim=2] cython_u(np.ndarray[double, ndim=2] distances):
    cdef np.ndarray[double, ndim=2] result = np.empty_like(distances)
    cdef double epsilon = 1e-6
    cdef int i, j

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            result[i, j] = distances[i, j] ** 2 * log(distances[i, j] + epsilon)

    return result