# Camera

# NOTE: to get the positive values in an array use array[array >= 0]:

import numpy as np
import pyrr

from image import Image
from ray import Ray
from functions import normalize, mag, distance, pos_min, arctan, arccos
from integrator import integrate_motion
from sphere import Sphere

# define image in raster space
# raster space -> NDC space
# NDC space -> camera space
# camera space -> world space

#texture_angle = 2*np.pi/(240000)
texture_angle = 2*np.pi/(36)

class Camera():
    def __init__(self, position, direction, width, height, fov, background_color = np.array([25, 0, 25])):
        self.pos = position
        self.dir = direction/np.linalg.norm(direction)
        self.width = width
        self.height = height
        self.depth = 1.0
        self.fov = fov
        self.aspect = width/height
        self.background_color = background_color
        
        print("Created Camera Successfully")
        
    def capture(self, scene, mode = "Minkowski"):
        """ saves the scene as an image file """
        
        # initialize image
        image = Image(self.width, self.height)
        
        # camera transformations
        translate = np.transpose(pyrr.Matrix44.from_translation(self.pos)) # pyrr matrices are opposite major and must be transposed
        #rotate = pyrr.Matrix44.identity # ignore for now
        #scale = pyrr.Matrix44.identity # ignore for now

        camera_to_world = translate #np.matmul(translate, np.matmul(rotate, scale))
        
        # for non-linear ray tracing:
        # Definition: "Sphere of Influence" will refer to the region
        #             in which space-time cannot be considered flat
        #             and ray paths must be integrated.
        
        global texture_angle
        
        soi_factor = 10 # sphere of influence factor
        soi_array = np.empty([scene.masses.shape[0]], dtype = Sphere)
        for i in range(scene.masses.shape[0]):
            soi_array[i] = Sphere(scene.masses[i].pos, scene.masses[i].r*soi_factor)
        
        for y in range(image.height):
            for x in range(image.width):   
                # RAY SETUP
                
                # raster space -> NDC space (normalized device coordinates)
                ndc_x = (x + 0.5) / self.width
                ndc_y = (y + 0.5) / self.height
                
                #  NDC space -> camera space
                screen_x = (2*ndc_x - 1)*self.aspect*np.tan(np.radians(self.fov/2))
                screen_y = (2*ndc_y - 1)*self.aspect*(-1) # inverted to correct image
                
                # create a ray in camera space
                # simplify transpose of ray position and ray direction vectors
                ray = Ray(np.transpose(np.array([screen_x, self.depth, screen_y])), np.transpose(np.array([screen_x, self.depth, screen_y])))
                
                # camera space -> world space
                ray.pos = np.array(np.matmul(camera_to_world, np.append(ray.pos, 1)))[:3] # transformation using homogenous coordinates
                ray.dir = np.array(np.matmul(camera_to_world, np.append(ray.dir, 0)))[:3] # zero to allow rotation but not translation
                
                # RAY TRACING
                if mode == "Minkowski":
                    # linear ray trace algorithm
                    
                    # find ray intersections and distances
                    t0_array = np.zeros([scene.masses.shape[0]])
                    for i in range(t0_array.shape[0]):
                        t0_array[i] = ray.check_intersection(scene.masses[i])
                    
                    # if all t0 are negative, there's no intersection
                    if np.all(t0_array < 0):
                        image.data[x][y] = self.background_color
                        continue
                    
                    # check for smallest t0 >= 0, closest takes render presidence
                    t0, index = pos_min(t0_array)
                    
                    mass = scene.masses[index]
                    
                    # lighting coefficient
                    I = ray.pos + t0*ray.dir # intersection point
                    N = normalize(I - mass.pos) # surface normal on sphere
                    lc = abs(np.dot(ray.dir, N)) # lighting coefficient
                    
                    # texture
                    # x = [0], z = [1], y = [2]
                    # sphere points in +y
                    sphere_theta = np.arccos((I-mass.pos)[2] / mass.r)
                    sphere_phi = arctan((I-mass.pos)[1], (I-mass.pos)[0])
                    
                    # texture_angle defined globally
                    
                    theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
                    phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
                    
                    if theta_condition ^ phi_condition: # XOR
                        image.data[x][y] = np.array(lc*mass.color1, dtype = np.int64)
                        continue
                    else:
                        image.data[x][y] = np.array(lc*mass.color2, dtype = np.int64)
                        continue
                
                if mode == "Schwarzschild":
                    # non-linear ray trace algorithm
                    
                    # position = ray.pos
                    # momentum = ray.dir
                    
                    # problems: 
                    # I expect this algorithm will fail if there is a mass
                    # within a sphere of influence being examined. add support for sphere detection
                    # to fix the problem when masses can't be used.
                    
                    # I expect that a mass with a sphere of influence radius that is smaller 
                    # than the mass's radius will not render the mass.
                    # but then how did the plane render with no mass implying rs = 0...
                    
                    # time
                    t = 0.0
                    dt = 0.3
                    #dt = 0.75
                    
                    #print("\n")
                    #print(" ===== pixel: x:{0}, y:{1} ===== ".format(x, y))
                    
                    # beware of break commands and how they will work in multiple embedded while loops
                    is_break = False
                    #while (is_break == False):
                    while (True):
                        # is the ray in a sphere of influence?
                        in_soi = False
                        for i in range(soi_array.shape[0]):
                            if ( distance(ray.pos, soi_array[i].pos) <= soi_array[i].r ):
                                in_soi = True
                                soi = soi_array[i]
                                index = i
                        
                        if (in_soi):
                            # find mass associated with sphere of influence
                            #print("in a sphere of influence. finding the associated mass")
                            mass = scene.masses[index]
                            
                            # integrate until escape or intersection
                            while (in_soi):
                                dX, dP = integrate_motion(t, ray.pos, ray.dir, mass.rs, dt, mode = "euler")
                                t0 = ray.check_intersection(mass)
                                #print("t0 = {0}".format(t0))
                                #print("magnitude of dX: {0}".format(mag(dX)))
                                #if t0 >= 0 and mag(dX) >= t0:
                                if (t0 > 0 and mag(dX) >= t0):
                                    # intersection
                                    I = ray.pos + t0*ray.dir # intersection point
                                    N = normalize(I - mass.pos) # surface normal on sphere
                                    lc = abs(np.dot(ray.dir, N)) # lighting coefficient
                                    
                                    # texture
                                    # x = [0], z = [1], y = [2]
                                    # sphere points in +z
                                    sphere_theta = np.arccos((I-mass.pos)[2] / mass.r)
                                    sphere_phi = arctan((I-mass.pos)[1], (I-mass.pos)[0])
                                    
                                    # texture_angle defined globally
                                    
                                    theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
                                    phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
                                    
                                    if (theta_condition ^ phi_condition): # XOR
                                        image.data[x][y] = np.array(lc*mass.color1, dtype = np.int64)
                                        #print("set color to color1")
                                        is_break = True
                                        break
                                    else:
                                        image.data[x][y] = np.array(lc*mass.color2, dtype = np.int64)
                                        #print("set color to color2")
                                        is_break = True
                                        break
                                else:
                                    # update ray
                                    ray.pos += dX
                                    ray.dir += dP
                                    
                                    # check if still in soi
                                    in_soi = distance(ray.pos, soi.pos) < soi.r
                                    
                                    #print("position: {0}".format(ray.pos))
                                    #print("direction: {0}".format(ray.dir))
                            
                            if (is_break):
                                #print("breaking from while loop")
                                break
                        
                        #print("successfully left in_soi loop without break")
                        # check if ray will intersect with soi
                        soi_t0_array = np.zeros([soi_array.shape[0]])
                        for i in range(soi_array.shape[0]):
                            soi_t0_array[i] = ray.check_intersection(soi_array[i])
                        
                        #is_soi_intersection = np.any(soi_t0_array < -1)
                        is_soi_intersection = np.any(soi_t0_array >= 0)
                        
                        t0_array = np.zeros([scene.masses.shape[0]])
                        for i in range(scene.masses.shape[0]):
                            t0_array[i] = ray.check_intersection(scene.masses[i])
                        is_mass_intersection = np.any(t0_array >= 0)
                        
                        if (is_soi_intersection):
                            #print("soi intersection. moving to soi")
                            soi_t0 = pos_min(soi_t0_array)[0]
                            ray.pos += soi_t0*ray.dir #+ 0.01*ray.dir # add a little extra to avoid floating point error
                        elif (is_mass_intersection):
                            t0, index = pos_min(t0_array)
                            
                            mass = scene.masses[index]
                            
                            # intersection
                            I = ray.pos + t0*ray.dir # intersection point
                            N = normalize(I - mass.pos) # surface normal on sphere
                            lc = abs(np.dot(ray.dir, N)) # lighting coefficient
                            
                            # texture
                            # x = [0], z = [1], y = [2]
                            # sphere points in +z
                            sphere_theta = np.arccos((I-mass.pos)[2] / mass.r)
                            sphere_phi = arctan((I-mass.pos)[1], (I-mass.pos)[0])
                            
                            # texture_angle defined globally
                            
                            theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
                            phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
                            
                            if (theta_condition ^ phi_condition): # XOR
                                image.data[x][y] = np.array(lc*mass.color1, dtype = np.int64)
                                #print("set color to color1")
                                is_break = True
                                break
                            else:
                                image.data[x][y] = np.array(lc*mass.color2, dtype = np.int64)
                                #print("set color to color2")
                                is_break = True
                                break
                            
                        else:
                            #print("no soi intersection")
                            image.data[x][y] = self.background_color
                            #print("set color to background color")
                            is_break = True
                            break
                    
                    #print("pixel successfully processed")
                        
                
            print("{:0.0f}%".format(y/self.height*100))
        print("100%")
        
        image.save()
        
        print("Captured Image Successfully")