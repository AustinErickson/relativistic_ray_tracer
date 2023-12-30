# Stores the equations of motion for a test particle in the Schwarschild metric

import numpy as np

# x[0] = x
# x[1] = y
# x[2] = z

def dx_dt(t, x, X, P, rs):
    r = np.sqrt(x**2 + X[1]**2 + X[2]**2)
    return (1+rs/(4*r))**(-6)*(1-rs/(4*r))**(2)*P[0]

def dy_dt(t, y, X, P, rs):
    r = np.sqrt(X[0]**2 + y**2 + X[2]**2)
    return (1+rs/(4*r))**(-6)*(1-rs/(4*r))**(2)*P[1]

def dz_dt(t, z, X, P, rs):
    r = np.sqrt(X[0]**2 + X[1]**2 + z**2)
    return (1+rs/(4*r))**(-6)*(1-rs/(4*r))**(2)*P[2]

def dpx_dt(t, px, X, P, rs):
    r = np.sqrt(np.dot(X, X))
    p_squared = np.dot(P, P)
    return -1/(2*r**3)*( (1-rs/(4*r))**(2)*(1+rs/(4*r))**(-7)*p_squared + (1-rs/(4*r))**(-1)*(1+rs/(4*r))**(-1) )*rs*X[0]

def dpy_dt(t, py, X, P, rs):
    r = np.sqrt(np.dot(X, X))
    p_squared = np.dot(P, P)
    return -1/(2*r**3)*( (1-rs/(4*r))**(2)*(1+rs/(4*r))**(-7)*p_squared + (1-rs/(4*r))**(-1)*(1+rs/(4*r))**(-1) )*rs*X[1]

def dpz_dt(t, pz, X, P, rs):
    r = np.sqrt(np.dot(X, X))
    p_squared = np.dot(P, P)
    return -1/(2*r**3)*( (1-rs/(4*r))**(2)*(1+rs/(4*r))**(-7)*p_squared + (1-rs/(4*r))**(-1)*(1+rs/(4*r))**(-1) )*rs*X[2]