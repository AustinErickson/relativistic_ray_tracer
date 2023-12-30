# Scene is a container for all objects (except the camera)

import numpy as np

class Scene():
    def __init__(self, masses = np.array([]), spheres = np.array([])):
        self.masses = masses
        self.spheres = spheres
