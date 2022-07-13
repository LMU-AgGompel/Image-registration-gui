# __init__.py

"""
ag_gompel_image_annotation_gui.

A graphical interface to annotate and place landmarks on images, built with the PySimpleGUI TKinter framework.

"""

__version__ = "0.1.0"
__author__ = 'Stefano Ceolin'


from image_registration.gui.gui import start_image_registration_GUI as start
from image_registration.registration.TPS import TPSwarping as TPSwarping
from image_registration.CNN.CNN import initialize_CNN, create_CNN, continue_CNN
