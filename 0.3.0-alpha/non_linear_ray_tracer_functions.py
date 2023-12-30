import numpy as np

from constants import soi_factor, background_color, dt
from geometric_tests import ray_sphere_intersection
from functions import integrate_schwarzschild, calculate_mass_surface_color

shaft_width_scale = 0.1
head_width_scale = 0.2

def outside_soi(ray_position, ray_direction, scene, ray_index, ray_count, mass_count, color_array):
    #print("------------outside soi------------")
    """ array initialization """
    mass_intersections = np.full([ray_count,3], np.array([np.nan, np.nan, np.nan]))
    mass_intersection_distances = np.full(mass_count, np.nan)
    
    soi_intersections = np.full([ray_count,3], np.array([np.nan, np.nan, np.nan]))
    soi_intersection_distances = np.full(mass_count, np.nan)
    
    """ calculate mass and soi intersections """
    for m in range(mass_count): # for every mass "m" (m = mass_index)
        # fill arrays with intersection points and distance values for the given ray and every mass in the scene
        mass_intersections[m], mass_intersection_distances[m] = ray_sphere_intersection(ray_position, ray_direction, scene.masses[m].position, scene.masses[m].radius)
        soi_intersections[m], soi_intersection_distances[m] = ray_sphere_intersection(ray_position, ray_direction, scene.masses[m].position, scene.masses[m].rs*soi_factor)
        
    """ setup to calculate closest intersection """
    # find the lowest positive t0 value. this value corresponds with the closest intersection.
    # first set all negative intersection distance values as +infinity so that the lowest non-negative one may be easily found.
    
    for i, element in enumerate(mass_intersection_distances):
        if element < 0 or np.isnan(element):
            mass_intersection_distances[i] = np.inf
            
    for i, element in enumerate(soi_intersection_distances):
        if element < 0 or np.isnan(element):
            soi_intersection_distances[i] = np.inf
    
    """ check whether mass or soi intersection occured """
    is_mass_intersection = not np.all(mass_intersection_distances == np.inf)
    is_soi_intersection = not np.all(soi_intersection_distances == np.inf)
    
    """ calculate the color or move ray to soi intersection """
    # NOTE: what if the closest distance is an array of two equal distances? this will likely crash the program.
    if (is_mass_intersection == False and is_soi_intersection == False):
        #print("is_mass_intersection == False and is_soi_intersection == False")
        color_array[ray_index] = background_color
        
    elif (is_mass_intersection == True and is_soi_intersection == False):
        #print("is_mass_intersection == True and is_soi_intersection == False")
        minimum_mass_distance = np.amin(mass_intersection_distances)
        minimum_mass_distance_index = np.argmin(mass_intersection_distances)
        mass = scene.masses[minimum_mass_distance_index]
        
        intersection_point = ray_position + minimum_mass_distance*ray_direction
        intersection_color = calculate_mass_surface_color(intersection_point, mass)
        
        color_array[ray_index] = intersection_color
        
    elif (is_mass_intersection == False and is_soi_intersection == True):
        #print("is_mass_intersection == False and is_soi_intersection == True")
        minimum_soi_distance_index = np.argmin(soi_intersection_distances)
        intersection_point = soi_intersections[minimum_soi_distance_index]
        
        ray_position = intersection_point
        
        mass = scene.masses[minimum_soi_distance_index]
        # take a tiny step so that the ray-sphere intersection doesn't fail at the boundary of the soi
        dx, dp = integrate_schwarzschild(ray_position, ray_direction, mass.position, mass.rs, dt)
        ray_position += dx
        ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
        
        mass_index = minimum_soi_distance_index
        
        inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)
        
    elif (is_mass_intersection == True and is_soi_intersection == True):
        #print("is_mass_intersection == True and is_soi_intersection == True")
        # determine which is closest
        
        # minimum mass distance
        minimum_mass_distance_index = np.argmin(mass_intersection_distances)
        minimum_mass_distance = mass_intersection_distances[minimum_mass_distance_index]
        #minimum soi distance
        minimum_soi_distance_index = np.argmin(soi_intersection_distances)
        minimum_soi_distance = soi_intersection_distances[minimum_soi_distance_index]
        
        # compare the two distances
        minimum_distances = np.array([minimum_mass_distance, minimum_soi_distance])
        minimum_distances_index = np.argmin(minimum_distances)
    
        # determine which one is the shortest distance
        if (minimum_distances_index == 0):
            #print("minimum_distances_index == 0")
            # the mass was intersected
            index = minimum_mass_distance_index
            
            intersection_point = mass_intersections[index]
            intersection_distance = minimum_mass_distance
            
            mass = scene.masses[minimum_mass_distance_index]
            
            intersection_color = calculate_mass_surface_color(intersection_point, mass)
            
            color_array[ray_index] = intersection_color
        else:
            #print("minimum_distances_index == 1")
            # the soi was intersected
            index = minimum_soi_distance_index
            
            intersection_point = soi_intersections[index]
            intersection_distance = minimum_soi_distance
            
            ray_position = intersection_point
            
            mass = scene.masses[index]
            # take a tiny step so that the ray-sphere intersection doesn't fail at the boundary of the soi
            dx, dp = integrate_schwarzschild(ray_position, ray_direction, mass.position, mass.rs, dt)
            ray_position += dx
            ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
            
            mass_index = index
            
            
            inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)
    
    
def inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array):
    """ this code assumes that there are no masses within the sphere of influence (except the central mass). """
    #print("------------inside soi------------")
    mass = scene.masses[mass_index]
    
    # integration for the following steps
    dx, dp = integrate_schwarzschild(ray_position, ray_direction, mass.position, mass.rs, dt)
    
    """ calculate mass and soi intersections """
    mass_intersection, mass_intersection_distance = ray_sphere_intersection(ray_position, ray_direction, mass.position, mass.radius)
    soi_intersection, soi_intersection_distance = ray_sphere_intersection(ray_position, ray_direction, mass.position, mass.rs*soi_factor)
        
    """ setup to calculate closest intersection """
    # find the lowest positive t0 value. this value corresponds with the closest intersection.
    # first set all negative intersection distance values as +infinity so that the lowest non-negative one may be easily found.
    
    if mass_intersection_distance < 0 or np.isnan(mass_intersection_distance):
        #print("no mass intersection")
        mass_intersection_distance = np.inf
            
    if soi_intersection_distance < 0 or np.isnan(soi_intersection_distance):
        #print("no soi intersection")
        soi_intersection_distance = np.inf
    
    """ check whether mass or soi intersection occured """
    is_mass_intersection = not (mass_intersection_distance == np.inf)
    is_soi_intersection = not (soi_intersection_distance == np.inf)
    
    """ calculate the color or move ray to soi intersection """
    # NOTE: what if the closest distance is an array of two equal distances? this will likely crash the program.
    if (is_mass_intersection == False and is_soi_intersection == False):
        outside_soi(ray_position, ray_direction, scene, ray_index, ray_count, mass_count, color_array)
        #raise Exception("No mass or soi intersection inside the soi. This should not be possible and something is broken...")
        # in theory this should be impossible, but what it is saying is that the little step that is
        # used to prevent soi boundary intersection issues has taken the ray out of the soi
    
    elif (is_mass_intersection == True and is_soi_intersection == False):
        #print("is_mass_intersection == True and is_soi_intersection == False")
        if (np.linalg.norm(dx) > mass_intersection_distance):
            #print("|dx| > mass_intersection_distance")
            
            intersection_point = ray_position + mass_intersection_distance*ray_direction
            
            intersection_color = calculate_mass_surface_color(intersection_point, mass)
            
            color_array[ray_index] = intersection_color
        else:
            #print("|dx| <= mass_intersection_distance")
            ray_position += dx
            ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
            inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)

    elif (is_mass_intersection == False and is_soi_intersection == True):
        #print("is_mass_intersection == False and is_soi_intersection == True")
        if (np.linalg.norm(dx) > soi_intersection_distance):
            #print("|dx| > soi_intersection_distance")
            ray_position = soi_intersection + dx
            ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
            outside_soi(ray_position, ray_direction, scene, ray_index, ray_count, mass_count, color_array)
        else:
            #print("|dx| <= soi_intersection_distance")
            ray_position += dx
            ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
            inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)
        
    elif (is_mass_intersection == True and is_soi_intersection == True):
        #print("is_mass_intersection == True and is_soi_intersection == True")
        # determine which is closest
        # compare the two distances
        distances = np.array([mass_intersection_distance, soi_intersection_distance])
        minimum_distance_index = np.argmin(distances)
        
        #print(distances)
        
        # determine which one is the shortest distance
        if (minimum_distance_index == 0):
            #print("minimum_distance_index == 0")
            # the mass was intersected
            if (np.linalg.norm(dx) > mass_intersection_distance):
                #print("|dx| > mass_intersection_distance")
                intersection_point = ray_position + mass_intersection_distance*ray_direction
                intersection_color = calculate_mass_surface_color(intersection_point, mass)
                
                color_array[ray_index] = intersection_color
            else:
                #print("|dx| <= mass_intersection_distance")
                ray_position += dx
                ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
                inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)
        else:
            #print("minimum_distance_index == 1")
            # the soi was intersected
            if (np.linalg.norm(dx) > soi_intersection_distance):
                #print("|dx| > soi_intersection_distance")
                ray_position = soi_intersection + dx
                ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
                outside_soi(ray_position, ray_direction, scene, ray_index, ray_count, mass_count, color_array)
            else:
                #print("|dx| <= soi_intersection_distance")
                ray_position += dx
                ray_direction += dp; ray_direction /= np.linalg.norm(ray_direction)
                inside_soi(ray_position, ray_direction, scene, ray_index, mass_index, ray_count, mass_count, color_array)