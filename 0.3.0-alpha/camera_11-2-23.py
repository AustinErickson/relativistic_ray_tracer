# NOTE: adding a random generation seed for stellar systems would be very cool.

import numpy as np
from functions import arctan, arccos, integrate_schwarzschild
import pyrr

from geometric_tests import ray_sphere_intersection
from mass import Mass
from scene import Scene
from image import Image

#testing:
import vpython as vp
def vec3(numpy_array):
    return vp.vec(numpy_array[0], numpy_array[1], numpy_array[2])

# move this to a constants file
background_color = np.array([255/2, 0, 255/2], dtype = np.int64)

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
        
        """
        re-examine code comments to make sure new implementation and comments don't conflict
        """
        
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
        ray_positions = np.vstack([X, Y, Z, W]).T # transpose to send column-major to row-major
        
        # why flatten it into a 1D array of np vec4 instead of leaving it as the more
        # intuitive 2D array of vec4 that would repesent the screen coordinates more geometrically?
        
        # the advantage is that we may now apply linear transformations to 
        # the list of arrays to transform them all at once.
        
        # this enables us to calculate the transformations individually, and
        # then create a singular combined matrix operation via matrix multiplication
        # before applying it to the array, saving computational expense.
        
        # more importantly, this frames the problem in terms of linear operations which makes
        # the problem much more intuitive and approachable.
        
        ### raster space
        # apply a translation to move the coordinates to the center of the pixels
        # pyrr transformations assume a row vector. this code uses column vectors and thus all transformations must be transposed.
        to_pixel_center = pyrr.matrix44.create_from_translation([0.5, 0.5, 0])
        
        ### normalized device coordinate space (NDC space)
        # apply scaling to normalize screen side lengths to the range [0, 1]
        to_ndc = pyrr.matrix44.create_from_scale([1/self.resolution[0], 1/self.resolution[1], 1])
        
        ### screen space
        # apply scaling so that sides range from [0, 2] 
        # apply translation so that side lengths are in the range [-1, 1]
        # apply mirror transformation to y to change the coordinate system so that y+ points up the screen and not down the screen (this is now possible since the range is from [-1, 1])
        
        scale = pyrr.matrix44.create_from_scale([2, 2, 1])
        translate = pyrr.matrix44.create_from_translation([-1, -1, 0])
        mirror_y = pyrr.matrix44.create_from_scale([1, -1, 1])
        
        # the '@' operator means "matmul"
        to_screen_space = scale @ translate @ mirror_y
        
        ### camera space
        # apply scaling to correct for fov and aspect ratio
        fov_correction = np.tan(np.radians(self.fov/2)) # store value in variable so that it only needs to be calculated once
        to_camera_space = pyrr.matrix44.create_from_scale([self.aspect_ratio*fov_correction, fov_correction, 1])
        
        ### to world
        # translate positions by camera position in world
        to_world = pyrr.matrix44.create_from_translation(self.position)
        
        ### ray directions and photon positions
        # ray directions are derived from the positions by using a lookat matrix and normalizing the resulting vectors
        # a lookat matrix orients the direction by taking the eye position (camera position in world space), 
        # a target position that one seeks to look at, and an arbitrary up vector used to form a basis through cross products.
        # since the camera basis relies on a cross product between the up vector and the target, the target and
        # up vector may not be pointing in the same direction, as the cross product will fail.
        
        lookat = pyrr.matrix44.create_look_at(self.position, self.target, self.up).T # only works when transposed. why?
        
        ### create camera to world matrix
        """ apply to_world after the positions are copied for ray directions """
        camera_to_world = to_pixel_center @ to_ndc @ to_screen_space @ to_camera_space @ np.linalg.inv(lookat.T)
        """ look_at is inverted because pyrr thinks objects are moving and not the camera? it is inverted for reasons I do not understand... """
        
        ### send ray positions to world space
        ray_positions = ray_positions @ camera_to_world
        
        ### generate directions from oriented and normalized ray positions
        #ray_directions = ray_positions*1 # copy position data to ray directions array
        #ray_directions = ray_positions*1
        
        ray_directions = ray_positions @ np.linalg.inv(to_world)
        
        # reduce from homogenous to standard cartesian coordinates
        ray_directions = ray_directions[:, 0:3]
        ray_positions = ray_positions[:, 0:3]
        
        # normalize the directions
        for i in range(ray_directions.shape[0]):
            ray_directions[i] = ray_directions[i] / np.linalg.norm(ray_directions[i])
        
        return ray_positions, ray_directions
    
    # public
    def capture(self):
        # add file_name and file_type support
        """ captures and saves the scene as an image file """
        
        """ Check for Invalid Program State """
        # check if there is a bound scene
        if (Scene.bound_scene == -1):
            # there is no bound scene
            raise Exception("No scene is bound. A scene must be bound to capture.")
        
        # raise exception if masses cannot be too close
        
        """ Initialization """
        # initialize rays
        ray_positions, ray_directions = self.initialize_rays()
        
        # initialize colors
        color_array = np.full(ray_positions.shape, [-1, -1, -1]) # set to an int it shouldn't be for dubugging
        
        # current bound scene
        scene = Scene.scenes[Scene.bound_scene]
        mass_count = scene.masses.shape[0] # number of masses
        
        """ There Are No Masses in the Scene """
        # check if there are even any masses in the scene
        if (mass_count == 0): # if the bound scene has no masses
            # no calculation needed. return the background color.
            color_array = np.full([self.resolution[0]*self.resolution[1], 3], np.full(3, background_color), dtype = np.int64)
        
        """ For Every Ray """
        for r in range(ray_positions.shape[0]): # for every ray "r":
            """ Progress Bar """
            # progress percentage
            print("{:.2f}".format(r/ray_positions.shape[0]*100), "%")
            
            """ Initialize Intersection and t0 Arrays """
            # make an array of ray intersection points and t0 values for every mass
            intersection_array = np.full(ray_positions.shape, np.array([np.nan, np.nan, np.nan]))
            t0_array = np.full(mass_count, np.nan)
            
            """ For Every Mass: Check Intersection Between Ray and Mass """
            for m in range(mass_count): # for every mass "m"
                # fill arrays with intersection points and t0 values for the given ray and every mass in the scene
                intersection_array[m], t0_array[m] = ray_sphere_intersection(ray_positions[r], ray_directions[r], scene.masses[m].position, scene.masses[m].radius)
                
            # find the lowest positive t0 value. this value corresponds with the closest intersection.
            # first set all negative t0 values as +infinity so that the lowest non-negative t0 may be easily found.
            
            """ Calculate Closest Intersection """
            for i, element in enumerate(t0_array):
                if element < 0 or np.isnan(element):
                    t0_array[i] = np.inf
            
            # if there are no positive (including zero) t0 values, there is no intersection
            if (np.all(t0_array == np.inf)):
                color_array[r] = background_color
                continue
            
            # calculate index of t0 corresponding with closest intersected mass
            index = np.argmin(t0_array)
            
            # store intersection point and t0
            intersection_point = intersection_array[index]
            t0 = t0_array[index]
            
            # the intersected mass
            mass = scene.masses[index]
            
            # the position and direction of the ray that intersected the mass
            ray_position = ray_positions[index]
            ray_direction = ray_directions[index]
            
            
            """ lighting """
            # simple color finding algorithm. change later
            
            #sphere_theta = np.arccos((surface_coordinate)[2] / radius)
            #sphere_phi = np.arctan((surface_coordinate)[1], (surface_coordinate)[0])
            
            # the position of the intersection in the mass' coordinates (same coordinates but translated so that the origin is th ecenter of the mass)
            surface_coordinate = intersection_point - mass.position
            surface_normal = surface_coordinate / np.linalg.norm(surface_coordinate)
            
            """ what is actually going on here? how is the ray direction not equivelent to intersection_point - camera_position? """
            direction_from_camera = intersection_point - self.position; direction_from_camera /= np.linalg.norm(direction_from_camera)
            #direction_from_camera = ray_direction - ray_position; direction_from_camera /= np.linalg.norm(direction_from_camera)
            #direction_from_camera = ray_position - self.position
            
            #vp.arrow(pos = vec3(ray_position), axis = vec3(direction_from_camera), color = vp.color.orange)
            
            lighting_coefficient = abs(np.dot(direction_from_camera, -surface_normal)) # surface normal lighting, flip surface normal so it points out of the surface.
            
            sphere_theta = arccos((surface_coordinate)[1] / mass.radius)
            if ((surface_coordinate[0] == 0) or (surface_coordinate[2] == 0)): 
                # for the exact poles, the checkered color pattern is problematic.
                # what color should the exact center of the pole have?
                # this reflects in the arctan to find phi.
                # what is phi when there is only a vertical component? at the poles it could be any angle.
                # just set angle to zero to prevent nan issues in the arctan function.
                sphere_phi = 0
            else:
                sphere_phi = arctan((surface_coordinate)[2], (surface_coordinate)[0])
            
            texture_angle = 2*np.pi / mass.checkered_subdivision
            
            theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
            phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
            
            if (theta_condition ^ phi_condition):
                color = np.array(lighting_coefficient * mass.color1, dtype = np.int64)
            else:
                color = np.array(lighting_coefficient * mass.color2, dtype = np.int64)
            
            # store color corresponding to ray
            color_array[r] = color
        
        # progress percentage
        print("{:.2f}".format(100), "%")
        
        # save color data to image file
        image = Image(self.resolution[0], self.resolution[1], color_array)
        image.save(file_type = 'ppm')
        