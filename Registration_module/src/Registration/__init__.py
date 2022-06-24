"""
Registration
=====

Provides function to realize image warping from detected landmarks and reference landmarks,
as well as further landmarks interpolation using edge detection.


Available subpackages
---------------------

thinplate
	Base of the thinplate spline mathematic model, used in the TPS module

TPS
	Allow the user to input an image and landmarks to warp it

snake
	Interpolate sliding landmark between existing ones using edge detection
"""

from Registration.thinplate import *
from Registration.snake import *
from Registration.TPS import *
try:
    import torch
    import thinplate.pytorch as torch
except ImportError:
    pass

__version__ = '1.0.0'
