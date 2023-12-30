# Handles 1st Order ODE Integration

import numpy as np

# optimized integration code
# Euler's Method: 

# I chose Euler's method because RK4 makes little difference, it is 4x as expensive, 
# and is much more complicated since the equations of motion are functions of the radial distance

# the equations of motion are independent of time and
# euler's method evaluates at only the current position
# which allows for large simplifications

def integrate_motion(ray, mass, dt):
    # 3-position, 3-momentum, mass position, and Scwarzschild radius
    x = ray.pos
    p = ray.dir
    x_m = mass.pos
    rs = mass.rs
    
    # displacement vector
    displacement = x - x_m
    #print("displacement: {0}".format(displacement))
    

    
    # radial distance, 3-momentum squared
    r = np.sqrt(np.dot(displacement, displacement))
    p_squared = np.dot(p, p)
    
    # equations of motion coefficients
    A = (1+rs/(4*r))**(-6)*(1-rs/(4*r))**(2)
    B = -1/(2*r**3)*( (1-rs/(4*r))**(2)*(1+rs/(4*r))**(-7)*p_squared + (1-rs/(4*r))**(-1)*(1+rs/(4*r))**(-1) )*rs
    
    # change in position, momentum
    dx = A*p*dt # dx/dt * dt
    dp = B*(x - x_m)*dt # dp/dt * dt # offset the position for non-orgin masses
    
    return dx, dp