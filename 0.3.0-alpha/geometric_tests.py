import numpy as np

def ray_sphere_intersection(ray_position, ray_direction, sphere_position, sphere_radius):
    """ takes a ray position, ray direction, sphere position, and sphere radius and returns the ray-sphere intersection point and t0 such that ray_position + t0*ray_direction = intersection_point """
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html
    # ray sphere intersection algorithm
    ray_to_sphere = sphere_position - ray_position # L
    tca = np.dot(ray_to_sphere, ray_direction) # tca
    
    """
    if (tca <= 0):
        # no intersection, ray is pointed away from sphere
        t0 = np.nan
        surface_coordinate = np.full([3], np.nan) # check dimensions of array later
        return surface_coordinate, t0
    """
    
    d_squared = np.dot(ray_to_sphere, ray_to_sphere) - tca**2 # d^2
    radius_squared = sphere_radius**2
    
    if (d_squared > radius_squared):
        # no intersection, ray will not intersect the sphere surface
        t0 = np.nan
        surface_coordinate = np.full([3], np.nan) # check dimensions of array later
        return surface_coordinate, t0
    
    
    
    # intersection
    thc = np.sqrt(radius_squared - d_squared)
    
    t0 = tca - thc
    t1 = tca + thc
    
    if (t0 > 0): # ray is outside the sphere
        intersection_point = ray_position + t0*ray_direction # intersection of ray and sphere in world space coordinates (origin is the world origin)
        return intersection_point, t0
    else: # the ray is inside the sphere
        # these are probably wrong
        #t0 = -(tca + thc)
        #t1 = -(tca - thc)
        
        # checking these ones based on worked out diagram of ray in sphere
        t0 = thc - tca
        t1 = thc + tca
        
        
        intersection_point = ray_position + t1*ray_direction
        return intersection_point, t1
    
    #surface_coordinate = intersection_point - sphere_position # intersection point of sphere surface in object space (origin is the sphere center)
    
    