# A general relativistic ray tracer for visualizing curved space-time.

# Supports compact masses which obey the Schwarzschild metric.
# Supports any number of compact masses sufficiently far from
# each other within one scene.

__author__ = "Austin Erickson"
__date__ = "12-30-23"
__version__ = "0.3.0-alpha"

import numpy as np

from datetime import datetime

from scene import Scene
from camera import Camera
from mass import Mass

def main():
    # time
    start_time = datetime.now()
    
    # define a scene
    scene = Scene()
    
    # create a camera
    camera = Camera(position = [0,0,10], target = [0, 0, -1], up = [0,1,0], resolution = [1080/4, 720/4], fov = 90.0)
    
    # create a mass
    center_mass = Mass(position = [0, 0, 0], radius = 2, mass = 0.5, color = [50, 225, 225], texture = 'checkered', checkered_subdivision = 12)
    
    # create a test mass
    test_mass = Mass(position = [-5, 0, -10], radius = 5, mass = 0, color = [230, 200, 50], texture = 'checkered', checkered_subdivision = 12)
    
    # capture scene
    camera.capture()
    
    # images are saved to the "Images" folder in the directory of "main"
    
    # time
    end_time = datetime.now()
    delta_time = end_time - start_time
    print("Time Elapsed: ", delta_time)

if __name__ == "__main__":
    main()
