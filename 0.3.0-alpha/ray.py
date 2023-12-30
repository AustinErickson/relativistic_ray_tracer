# Handles Rays and Intersection

import numpy as np

from functions import normalize

class Ray():
    def __init__(self, position, direction):
        self.pos = position
        self.dir = normalize(direction)
    
    # assure that the direction is always normalized
    @property
    def dir(self):
        return self._dir
    
    @dir.setter
    def dir(self, direction):
        self._dir = normalize(direction)
    
    def check_intersection(self, sphere):
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        """ checks ray-sphere intersection \n
        returns t0 (ray.pos + t0*ray.dir = intersection point) or -1 if no intersection """
        L = sphere.pos - self.pos
        tca = np.dot(L, self.dir)
        
        if (tca < 0):
            return -1
        
        r_squared = sphere.r**2
        d_squared = np.dot(L, L) - tca**2
        
        if (r_squared < d_squared):
            return -1
        
        thc = np.sqrt(r_squared - d_squared)
        t0 = tca - thc
        
        if (t0 < 0):
            return -1
        
        return t0