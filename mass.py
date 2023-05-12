# Handles massive objects

import numpy as np

from sphere import Sphere

class Mass(Sphere):
    def __init__(self, position, radius, mass, color = np.array([255, 255, 255]), texture = "solid", checkered_subdivision = 24):
        super().__init__(position, radius, color, texture, checkered_subdivision)
        self.M = mass
        self.rs = 2*mass
        
        # remove later
        #self.color1 = 0.8*color
        #self.color2 = 0.2*color
        
        self.color1 = color
        self.color2 = color
