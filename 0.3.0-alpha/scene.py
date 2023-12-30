# a scene is a container for masses
# consider removing for simplicity

import numpy as np

class Scene():
    scenes = np.array([])
    bound_scene = -1
    
    def __init__(self, masses = np.array([])):
        self.masses = masses
        # append the scene to the scenes array
        Scene.scenes = np.append(Scene.scenes, self)
        
        # bind the generated scene
        # find the index of the scene in the scenes array:
        self._index = self.scenes.shape[0] - 1
        # bind the scene
        self.bind()
        
    
    def bind(self):
        Scene.bound_scene = self._index
        
    def delete(self):
        # delete scene and free memory
        del self.masses
        del self
    
    """
    def generate_random_mass_field():
        pass
    """