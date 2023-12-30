# Conveniance Functions

import numpy as np

def normalize(vector):
    """ returns a unit vector which points in the direction of the given vector 
        returns the zero vector for vectors with zero magnitude """
    mag = np.linalg.norm(vector)
    if (mag > 0):
        return vector / mag
    else:
        return np.array([0, 0, 0])

def mag(vector):
    """ returns the magnitude of a vector """
    return np.linalg.norm(vector)

def distance(position1, position2):
    """ returns the coordinate distance between to positions """
    return mag(position1 - position2)

def pos_min(array):
    """ returns the minimum positive entry in a 1D scalar array and its index (including zero) """
    pos_min = array[0]
    index = 0 # store index for querry of masses
    for i in range(array.shape[0]):
        if pos_min < 0:
            pos_min = array[i]
            index = i
            continue
        if (array[i] >= 0 and array[i] < pos_min):
            pos_min = array[i]
            index = i
    
    return pos_min, index

def arctan(y, x):
    """ returns the true ccw angle from the x-axis"""
    if (x > 0 and y >= 0):
        return np.arctan(y/x)
    if (x < 0):
        return np.arctan(y/x) + np.pi
    if (x > 0 and y < 0):
        return np.arctan(y/x) + 2*np.pi
    if (x == 0 and y > 0):
        return np.pi/2
    if (x == 0 and y < 0):
        return 3*np.pi/2
    if (x == 0 and y == 0):
        return np.nan
    if (x == 0):
        return np.nan
    
def arccos(x):
    """ returns np.arccos but assures the value passed is rounded to fit within the domain of arccos """
    # can you redefine a parameter that is passed in?
    if (x < -1):
        x = -1
    if (x > 1):
        x = 1
    return np.arccos(x)
