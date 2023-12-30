# Handles massive objects

import numpy as np

class Mass():
    def __init__(self, position, radius, mass, color = np.array([255, 255, 255])):
        self.pos = position
        self.r = radius
        self.M = mass
        self.rs = 2*mass
        self.color1 = 0.8*color
        self.color2 = 0.2*color