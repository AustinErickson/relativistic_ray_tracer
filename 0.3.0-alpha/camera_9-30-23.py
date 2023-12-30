# NOTE: adding a random generation seed for stellar systems would be very cool.

import numpy as np
from functions import normalize, arctan
import pyrr

from geometric_tests import ray_sphere_intersection
from mass import Mass
from image import Image

# problem:
# it is techincally possible for a mass's surface to be within another mass's soi. thus, for non-linear ray tracing,
# either make it a condition that this be an impossible case, or run a sort and find all mass's who's
# surfaces intersect the soi and add them to a list to check all of them for intersection during integration.

class Camera():
    """ 
    A camera generates images of a scene.
    
    members:
    + position : np vec3
    + target : np vec3
    + up : np vec3
    + resolution : np vec2
    + fov : double [degrees]
    - aspect_ratio : double
    - screen_depth : const double
    
    methods:
    - initialize_rays() => ray_position : np vec3, ray_direction : vec3
    - ray_sphere_intersection (ray_position, ray_direction, sphere_position, sphere_radius) => t0 : double, surface_coordinate : np vec3
    + capture(scene : scene object, ray_positions : np array of np vec3, ray_directions : np array of np vec3, mode : string)
    """
    def __init__(self, **kwargs):
        # members
        # public
        self.position = np.array(kwargs['position'])
        self.target = np.array(kwargs['target'])
        self.up = np.array(kwargs['up'])
        self.resolution = np.array(kwargs['resolution'], dtype = np.int64)
        self.fov = kwargs['fov']
        
        # private
        self.aspect_ratio = kwargs['resolution'][0] / kwargs['resolution'][1]
        self.screen_depth = 1.0
        
    #methods
    # private
    def initialize_rays(self):
        """ returns a numpy array of photon ray positions and photon ray directions for every pixel """
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays
        
        # conventions:
        # graphics libraries have the following conventions:
        # 1) the z-axis is depth, and the +y direction is up. this is because the screen coordinates are defined with x and y, leaving z to be depth.
        # 2) the camera points in the -z direction by default. this is a natural consequence of using the x and y axes to define the screen coordinate system from the bottom left corner (z-axis is x cross y => -z points in to the screen))
        # 3) *this contradicts with 2). fix* screen coordinates are read top to bottom, left to right, starting at the top left and ending at the bottom right. thus, +x points to the right of the screen and +y points down the screen.
        # 4) matrices and vectors are column major. this means vectors are columns, and matrix operation order is from right to left, with the vector being far right (math convention).)
        
        ### "array space"
        # define the x and y axis values
        x = np.arange(0, self.resolution[0], 1)
        y = np.arange(0, self.resolution[1], 1)
        
        # make a 2D set of coordinates from the axis values
        X, Y = np.meshgrid(x, y)
        
        # combine into a numpy array of np vec2
        # seperate the x and y coordinate values
        X = X.flatten()
        Y = Y.flatten()
        Z = np.full(X.shape, -self.screen_depth) # by convention, the camera points in -z
        W = np.full(X.shape, 1)
        # w is the homogenous coordinate and is set to 1. this 4th dimension is necessary to do translation transformations as linear transformations.
        
        # combine the x and y coordinate values into one np array of np vec2 ray positions
        ray_positions = np.vstack([X, Y, Z, W])
        
        # why flatten it into a 1D array of np vec3 instead of leaving it as the more
        # intuitive 2D array that would repesent the screen coordinates more geometrically?
        
        # the advantage is that we may now apply linear transformations to 
        # the list of arrays to transform them all at once.
        
        # this enables us to calculate the transformations individually, and
        # then create a singular combined matrix operation via matrix multiplication
        # before applying it to the array, saving computational expense.
        
        # this also frames the problem in terms of linear operations which makes
        # the problem much more intuitive and approachable.
        
        ### raster space
        # apply a translation to move the coordinates to the center of the pixels
        # pyrr transformations assume a row vector. this code uses column vectors and thus all transformations must be transposed.
        to_pixel_center = pyrr.matrix44.create_from_translation([0.5, 0.5, 0]).T
        
        ### normalized device coordinate space (NDC space)
        # apply scaling to normalize screen side lengths to the range [0, 1]
        to_ndc = pyrr.matrix44.create_from_scale([1/self.resolution[0], 1/self.resolution[1], 1]).T
        
        ### screen space
        # apply scaling so that sides range from [0, 2] 
        # apply translation so that side lengths are in the range [-1, 1]
        # apply mirror transformation to y to change the coordinate system so that y+ points up the screen and not down the screen (this is now possible since the range is from [-1, 1])
        
        scale = pyrr.matrix44.create_from_scale([2, 2, 1]).T
        translate = pyrr.matrix44.create_from_translation([-1, -1, 0]).T
        mirror_y = pyrr.matrix44.create_from_scale([1, -1, 1]).T
        
        # the '@' operator means "matmul"
        to_screen_space = (mirror_y @ translate @ scale)
        
        ### camera space
        # apply scaling to correct for fov and aspect ratio
        fov_correction = np.tan(np.radians(self.fov/2)) # store value in variable so that it only needs to be calculated once
        to_camera_space = pyrr.matrix44.create_from_scale([self.aspect_ratio*fov_correction, fov_correction, 1]).T
        
        
        ### send rays to camera space
        ray_positions = (to_camera_space @ to_screen_space @ to_ndc @ to_pixel_center @ ray_positions).T
        ### generate rays from normalized positions in camera space
        ray_directions = normalize(ray_positions[:, 0:3]) # normalize 3D part, then make homogenous again
        ray_directions = np.hstack( (ray_directions, np.full([ray_directions.shape[0], 1], 1)))
        
        ### send rays to world space
        #camera_to_world = np.linalg.inv(pyrr.matrix44.create_look_at(self.position, self.target, self.up))
        #camera_to_world = np.linalg.inv(pyrr.matrix44.create_look_at(self.position, self.target, self.up).T)
        
        ray_positions_to_world = pyrr.matrix44.create_from_translation(self.position).T
        
        ray_directions_to_world = pyrr.matrix44.create_look_at([0,0,0], self.target, self.up).T
        print(ray_directions_to_world)
        
        ray_positions = ray_positions_to_world @ ray_positions.T
        ray_directions = ray_directions_to_world @ ray_directions.T
        
        ray_positions = ray_positions.T # turn back into row vectors
        ray_positions = ray_positions[:, 0:3] # reduce to 3D
        
        ray_directions = ray_directions.T # turn back into row vectors
        ray_directions = ray_directions[:, 0:3] # reduce to 3D
        
        return ray_positions, ray_directions
    
    # public
    def capture(self):
        # add file_name and file_type support
        """ captures and saves the scene as an image file """
        ### camera to world space transformation
        
        # the below idea is problematic. it works for translations, but handling rotations become very complicated.
        """
        # note: instead of moving the camera, we will move the world.
        #       therefore, we will apply the inverse transformations to the scene objects.
        """
        
        #camera_to_world = pyrr.matrix44.create_look_at(self.position, self.target, self.up)
        
        """
        # transform masses' positions with camera_to_world transformation
        # for (mass in Mass.masses):
            # make list of positions
            
        # transform the positions all at once
        
        # for (mass in Mass.masses):
            # update mass positions
        """
        
        # initialize rays
        ray_positions, ray_directions = self.initialize_rays()
        
        # initialize arrays
        t0_array = np.full(len(Mass.masses), np.nan)
        surface_coordinate_array = np.full(ray_positions.shape, 0.0)
        color_array = np.full(ray_positions.shape, 0)
        
        """
        # HUGE PROBLEM: updating the mass positions directly will compund if you capture multiple times!
        #               make sure to not update the positions directly!
        # transform the mass positions
        for mass in Mass.masses:
        # for every mass
        
            # transform "the camera" (move world objects oppositely)
            #mass.position = (camera_to_world @ np.append(mass.position, 1))[0:3]
            
        # BIG PROBLEM: if there are no masses in the scene, the camera will break
        """
        # ray trace
        for i in range(ray_positions.shape[0]):
        # for every ray:
            # temp
            m = 0
            for mass in Mass.masses:
            # for every mass
                # calculate intersection(s)
                t0, surface_coordinate = ray_sphere_intersection(ray_positions[i], ray_directions[i], mass.position, mass.radius)
                t0_array[m] = t0
                surface_coordinate_array[m] = surface_coordinate
                
                # mass index
                m += 1
                
            # find the first mass intersection     
            t0 = np.nanmin(t0_array) # minimum t0 (ignoring nan, outputs single number, not array)
            if not (np.isnan(t0)): # it can still be nan if all are nan
                # if the number is not nan:
                m = np.array(np.where(t0_array == t0)).flat[0] # take the first redundant mass of intersection
                surface_coordinate = surface_coordinate_array[m]
                ray_direction = ray_directions[m]
                
                # simple color finding algorithm. change later
                # lighting with cieling light
                lighting_coefficient = 1
                
                #sphere_theta = np.arccos((surface_coordinate)[2] / radius)
                #sphere_phi = np.arctan((surface_coordinate)[1], (surface_coordinate)[0])
                
                sphere_theta = np.arccos((surface_coordinate)[1] / Mass.masses[m].radius)
                sphere_phi = arctan((surface_coordinate)[2], (surface_coordinate)[0])
                
                texture_angle = 2*np.pi / Mass.masses[m].checkered_subdivision
                
                theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
                phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
                
                if (theta_condition ^ phi_condition):
                    return_color = np.array(lighting_coefficient*Mass.masses[m].color1, dtype = np.int64)
                else:
                    return_color = np.array(lighting_coefficient*Mass.masses[m].color2, dtype = np.int64)
                
                color_array[i] = return_color
            else:
                # if the number is nan
                # no mass was intersected. return the background color.
                color_array[i] = np.array([100, 0, 100], dtype = np.int64)
        
        # send color data to an image object
        image = Image(self.resolution[0], self.resolution[1], color_array)
        image.save(file_type = 'ppm')
        