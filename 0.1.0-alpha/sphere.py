# Handles spheres (Masses with no mass)

# make Mass class child of this class?

import numpy as np

class Sphere():
    def __init__ (self, position, radius, color = np.array([255, 255, 255])):
        self.pos = position
        self.r = radius
        self.color = color