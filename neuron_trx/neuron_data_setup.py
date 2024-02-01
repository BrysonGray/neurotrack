#!/usr/bin/env python

'''

Functions for setting up neuron images and ground truth data in swc format for neurite tracking.

Author: Bryson Gray
2024

'''

# TODO: make sure these functions deal with the large img array efficiently
def draw_line_segment(img, segment):
    """ Draw a line segment with width.

    Parameters
    ----------
    img : Three dimensional scalar-valued array

    segment : list of points, each a tuple of x-y-z cartesian coordinates and radius (x,y,z,r)

    """

    pass

def draw_neurite_tree(img, segments):
    """ Draw all segments to reconstruct a whole neurite tree.

    Parameters
    ----------
    img : Three dimensional scalar-valued array

    segments : N x 2 x 4 array. Array of N segments, each consisting of two points, each point defined by a cartesian coordinate and radius (X,Y,Z,R).

    """

    pass

class DataLoader():
    
    def __init__(self, image : str, label : str) -> None:
        pass