# Handles spheres

import numpy as np

class Sphere():
    """ texture: solid, checkered """
    def __init__(self, position, radius, color = np.array([255, 255, 255]), texture = "solid", checkered_subdivision = 16):
        self.pos = position
        self.r = radius
        self.color = color
        self.texture = texture
        self.checkered_subdivision = checkered_subdivision