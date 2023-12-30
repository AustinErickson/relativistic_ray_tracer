# Handles 1st Order ODE Integration

# make this method cleaner later

import numpy as np

from motion import dx_dt, dy_dt, dz_dt, dpx_dt, dpy_dt, dpz_dt

def euler(dx_dt, t, x, X, P, rs, dt):
    """ 1st order integrator """
    return dx_dt(t, x, X, P, rs)*dt

def RK4(dx_dt, t, x, X, P, rs, dt):
    """ 4th order integrator """
    k1 = dx_dt(t, x,                    X, P, rs)
    k2 = dx_dt(t + dt/2, x + k1/2*dt,   X, P, rs)
    k3 = dx_dt(t + dt/2, x + k2/2*dt,   X, P, rs)
    k4 = dx_dt(t + dt,   x + k3*dt,     X, P, rs)
    
    return (1/6)*(k1 + 2*k2 + 2*k3 + k4)*dt

def integrate_motion(t, X, P, rs, dt, mode = "euler"):
    """ takes the time 3-position, 3-momentum and Schwarzschild radius
    and returns the change in position dx, and change in momentum dp """
    
    if mode == "RK4":
        # RK4
        dx  = RK4(dx_dt,  t, X[0], X, P, rs, dt)
        dy  = RK4(dy_dt,  t, X[1], X, P, rs, dt)
        dz  = RK4(dz_dt,  t, X[2], X, P, rs, dt)
        
        dpx = RK4(dpx_dt, t, P[0], X, P, rs, dt)
        dpy = RK4(dpy_dt, t, P[1], X, P, rs, dt)
        dpz = RK4(dpz_dt, t, P[2], X, P, rs, dt)
    
    if mode == "euler":
        # Euler
        dx  = euler(dx_dt,  t, X[0], X, P, rs, dt)
        dy  = euler(dy_dt,  t, X[1], X, P, rs, dt)
        dz  = euler(dz_dt,  t, X[2], X, P, rs, dt)
        
        dpx = euler(dpx_dt, t, P[0], X, P, rs, dt)
        dpy = euler(dpy_dt, t, P[1], X, P, rs, dt)
        dpz = euler(dpz_dt, t, P[2], X, P, rs, dt)
    
    return np.array([dx, dy, dz]), np.array([dpx, dpy, dpz])