# Scene is a container for all objects

import numpy as np

class Scene():
    def __init__(self, masses = np.array([])):
        self.masses = masses
