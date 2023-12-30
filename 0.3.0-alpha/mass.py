# Handles mass objects

import numpy as np
from scene import Scene

from constants import soi_factor

class Mass():
    def __init__(self, **kwargs):
        self.position = np.array(kwargs['position'])
        self.radius = kwargs['radius']
        self.mass = kwargs['mass']
        self.rs = 2*self.mass
        
        # set mass color and texture
        self.color = np.array(kwargs['color'], dtype = np.int64)
        self.texture = kwargs['texture']
        self.checkered_subdivision = kwargs['checkered_subdivision'] # implement later
        
        if (self.texture == 'solid'):
            self.color1 = self.color
            self.color2 = self.color
        elif (self.texture == 'checkered'):
            self.color1 = self.color // (5/4)
            self.color2 = self.color // (5/1)
        else:
            # display error color
            self.color1 = np.array([255, 0, 255], dtype = np.int64)
            self.color2 = self.color1
        
        # add mass to bound scene
        if (Scene.bound_scene == -1):
            raise Exception("A Scene Must Be Bound Before Masses May Be Initialized")
        
        masses = Scene.scenes[Scene.bound_scene].masses
        
        # check that the added mass and its soi does not intersect or touch any other mass or soi
        # (assure masses are sufficiently far so that the space-time between them is sufficiently flat)
        
        max_radius1 = max(self.rs*soi_factor, self.radius)
        
        masses_too_close = False
        
        for m in range(len(masses)):
            max_radius2 = max(masses[m].rs*soi_factor, masses[m].radius)
            
            seperation_distance = np.linalg.norm(self.position - masses[m].position)
            
            masses_too_close = max_radius1 + max_radius2 >= seperation_distance
        
            if (masses_too_close):
                raise Exception("A Mass-Mass, SOI-SOI, or Mass-SOI Intersection Has Occured. Masses must be sufficiently distance such that the masses and or their spheres of influence are not touching or intersecting.")
        
        Scene.scenes[Scene.bound_scene].masses = np.append(Scene.scenes[Scene.bound_scene].masses, self)
