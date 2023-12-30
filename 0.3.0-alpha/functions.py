# Conveniance Functions

import numpy as np
    
# conveniance functions (remove later? or implement for ease of legability?)
def mag(vector):
    """ returns the magnitude of a vector """
    return np.linalg.norm(vector)

def distance(position1, position2):
    """ returns the coordinate distance between to positions """
    return mag(position1 - position2)

"""
def surface_color(sphere, intersection_point):
    
    N = normalize(intersection_point - sphere.pos) # surface normal on sphere
    lc = 1.0 #abs(np.dot(ray.dir, N)) # lighting coefficient
    
    # texture
    # x = [0], z = [1], y = [2]
    # sphere points in +y
    sphere_theta = np.arccos((intersection_point-sphere.pos)[2] / sphere.r)
    sphere_phi = arctan((intersection_point-sphere.pos)[1], (intersection_point-sphere.pos)[0])
    
    texture_angle = 2*np.pi / sphere.checkered_subdivision
    
    theta_condition = (int(sphere_theta / texture_angle) % 2 == 0)
    phi_condition = (int(sphere_phi / texture_angle) % 2 == 0)
    
    if (theta_condition ^ phi_condition): # XOR
        color = np.array(lc*sphere.color1, dtype = np.int64)
        
    else:
        color = np.array(lc*sphere.color2, dtype = np.int64)
        
    return color
"""


# trig functions
def arctan(y, x):
    """ returns the true ccw angle from the x-axis """
    if (x > 0 and y >= 0):
        return np.arctan(y/x)
    elif (x < 0):
        return np.arctan(y/x) + np.pi
    elif (x > 0 and y < 0):
        return np.arctan(y/x) + 2*np.pi
    elif (x == 0 and y > 0):
        return np.pi/2
    elif (x == 0 and y < 0):
        return 3*np.pi/2
    elif (x == 0 and y == 0):
        return np.nan
    elif (x == 0):
        return np.nan
    else:
        return np.nan # just to make sure there isn't a case that can crash the program
    
def arccos(x):
    """ returns np.arccos but assures the value passed is rounded to fit within the domain of arccos """
    # can you redefine a parameter that is passed in?
    if (x < -1):
        x = -1
    if (x > 1):
        x = 1
    return np.arccos(x)

# integrator
def integrate_schwarzschild(ray_position, ray_direction, mass_position, schwarzschild_radius, dt):
    """ Takes a ray position, ray direction, mass position, Schwarzschild radius of the mass and a timestep dt.
    Returns the infinitesimal change in ray position and direction. """
    
    # NOTE: this needs to be changed to using momentum and not direction so it is more accurate. it is correct, but for the wrong reason.
    
    # 3-position, 3-momentum (anaolgous to direction), Schwarzschild radius
    x = ray_position - mass_position
    p = ray_direction
    rs = schwarzschild_radius
    
    # radial distance, 3-momentum squared
    r = np.sqrt(np.dot(x, x))
    p_squared = np.dot(p, p)
    
    # equations of motion coefficients (for conveniance)
    A = (1+rs/(4*r))**(-6)*(1-rs/(4*r))**(2)
    B = -1/(2*r**3)*( (1-rs/(4*r))**(2)*(1+rs/(4*r))**(-7)*p_squared + (1-rs/(4*r))**(-1)*(1+rs/(4*r))**(-1) )*rs
    
    # change in position, momentum
    dx = A*p*dt # dx/dt * dt
    dp = B*x*dt # dp/dt * dt
    
    return dx, dp

# gravitational wave metric

# plus mode function
def fplus(t, z, a, w):
    return a*np.sin(w*(t-z))

def dfplus_dt(t, z, a, w):
    return a*w*np.cos(w*(t-z))

def dfplus_dz(t, z, a, w):
    return -a*w*np.cos(w*(t-z))


def integrate_gravitational_wave(four_position, four_momentum, dt, a = 0.001, w = 0.001):
    """ Integrates a light ray in the gravitational wave metric for plus mode pularization """
    
    # position
    t = four_position[0]
    x = four_position[1]
    y = four_position[2]
    z = four_position[3]
    
    # momentum
    pt = four_momentum[0]
    px = four_momentum[1]
    py = four_momentum[2]
    pz = four_momentum[3]
    
    # space derivatives
    # dt_dt = 1
    dx_dt = -(1 - fplus(t, z, a, w)) * (px/pt)
    dy_dt = -(1 + fplus(t, z, a, w)) * (py/pt)
    dz_dt = -(pz/pt)
    
    # momentum derivatives
    dpt_dt = (1/2)*dfplus_dt(t, z, a, w) * ((px**2 - py**2) / pt)
    # dpx_dt = 0
    # dpy_dt = 0
    dpz_dt = (1/2)*dfplus_dz(t, z, a, w) * ((px**2 - py**2) / pt)
    
    # integrate position
    dx = dx_dt*dt
    dy = dy_dt*dt
    dz = dz_dt*dt
    
    x += dx
    y += dy
    z += dz
    
    # integrate momentum
    dpt = dpt_dt*dt
    dpz = dpz_dt*dt
    
    pt += dpt
    pz += dpz
    
    # this is likely incorrect since px and py don't change to account for pz's change
    four_position += np.array([dt, dx, dy, dz])
    four_momentum += np.array([dpt, 0, 0, dpz])
    
    return four_position, four_momentum
    

# simple color finding algorithm. change later

def calculate_mass_surface_color(intersection_point, mass):

    # the position of the intersection in the mass' coordinates (same coordinates but translated so that the origin is the center of the mass)
    surface_coordinate = intersection_point - mass.position
    
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
        color = np.array(mass.color1, dtype = np.int64)
    else:
        color = np.array(mass.color2, dtype = np.int64)
    
    return color