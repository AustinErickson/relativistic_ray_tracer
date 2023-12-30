# General Relatavistic Ray Tracer for Generating Images of Stellar Systems
__author__ = "Austin Erickson"
__date__ = "12-28-22"
__version__ = "0.2.0-alpha"

import numpy as np

from scene import Scene
from mass import Mass
from camera import Camera

# Debug:
# ray tracer only works if it start in sphere of influence?

# Eureka: The equations of motion think the singularity is at the origin.
#         update the equations to be generalized for any location.

# DO NOT forget to change the soi code back to rs*soi_factor

def main():
    # initialize masses
    #plain = Mass(np.array([0.0, 0.0, -100015.0]), 100000, 0.0, np.array([255, 0, 0]))
    #sphere1 = Mass(np.array([0.0, 15.0, 0.0]), 2.5, 0.0, np.array([250, 250, 0]))
    #blackhole = Mass(np.array([0.0, 0.0, 0.0]), 1.0, 0.5, np.array([255, 255, 255]))
    neutron_star = Mass(np.array([0.0, 0.0, 0.0]), 1.5, 0.5, np.array([255, 255, 255]))
    #masses = np.array([blackhole])
    
    masses = []
    masses.append(neutron_star)
    
    for y in range(8):
        for x in range(8):
            z = np.random.uniform(105.0, 125.0, 1)
            masses.append(Mass(np.array([8*(x-np.random.uniform(3.5, 4.5)), z, 8*(y-np.random.uniform(3.5, 4.5))]), 1.0, 0.0, np.array([255, 255, 255])))
    
    masses = np.array(masses)
    
    # initialize scene
    scene = Scene(masses)
    
    size = 100
    
    # initialize camera
    camera = Camera(np.array([0.0, -75.0, 0.0]), np.array([0.0, 1.0, 0.0]), size, size, 90.0, background_color = np.array([0, 0, 0]))
    
    # capture scene
    camera.capture(scene, mode = "Minkowski")
    #camera.capture(scene, mode = "Schwarzschild")

if __name__ == "__main__":
    main()
