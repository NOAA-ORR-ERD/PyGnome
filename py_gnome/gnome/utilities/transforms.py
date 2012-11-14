import numpy as np
import math

def r_theta_to_uv_current(r_theta):
    """
    Converts array of current values given with magnitude, direction into (u,v) wind values.
        Current from 0deg is (u,v) = (0,1), from 45deg (u,v) = (1,1)
        Basically, rotate clockwise from North. In addition, (u,v) represents the direction 
        the wind blows towards
    
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding uv cartesian velocity vector
    """
    xform = np.matrix([(-1., 0.), (0.,-1.)])
    uv = np.dot( r_theta_to_uv_wind(r_theta).view(dtype=np.matrix), xform)
    return np.asarray(uv)

def uv_to_r_theta_current(uv):
    """
    Converts array of current values given with (u,v) wind values to magnitude, direction.
        Current from 0deg is (u,v) = (0,1), from 45deg (u,v) = (1,1)
        Basically, rotate clockwise from North. In addition, (u,v) represents the direction 
        the wind blows towards
    
    :param uv: NX2 numpy array, where each row corresponds with a velocity vector 
    :returns: NX2 numpy array containing polar coordinates r_theta 
    """
    xform = np.matrix([(-1., 0.), (0.,-1.)])
    uv = np.dot( np.matrix(uv.reshape(-1,2)), xform)
    return uv_to_r_theta_wind(uv)

def r_theta_to_uv_wind(r_theta):
    """
    Converts array of wind values given with magnitude, direction into (u,v) wind values.
        Wind from 0deg is (u,v) = (0,-1), from 45deg (u,v) = (-1,-1)
        Basically, rotate clockwise from North. In addition, (u,v) represents the direction 
        the wind blows towards
    
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding uv cartesian velocity vector
    """
    r_theta = np.asarray(r_theta).reshape(-1,2)
    if np.any(r_theta[:,1] > 360) or np.any(r_theta[:,1] < 0):
        raise ValueError("input angle in r_theta[:,1] must be between 0 and 360")
    
    if np.any(r_theta[:,0] <= 0):
        raise ValueError("input magnitude in r_theta[:,0] must be greater than 0")
    
    rq = np.array(r_theta) 
    rq[:,1] = np.deg2rad(rq[:,1])
    
    uv = np.zeros_like(rq)
    uv[:,0] = np.round( rq[:,0]*np.sin(rq[:,1]), decimals=14)
    uv[:,1] = np.round( rq[:,0]*np.cos(rq[:,1]), decimals=14)
    
    # create matrix so -1*0 = 0 and not -0 and let's not screw up original
    uv = np.asarray( -1*uv.view(dtype=np.matrix) )
    
    return uv

def uv_to_r_theta_wind(uv):
    """
    Converts array of wind values given with (u,v) wind values to magnitude, direction.
        Wind from 0deg is (u,v) = (0,-1), from 45deg (u,v) = (-1,-1)
        Basically, rotate clockwise from North. In addition, (u,v) represents the direction 
        the wind blows towards
    
    :param uv: NX2 numpy array, where each row corresponds with a velocity vector 
    :returns: NX2 numpy array containing polar coordinates r_theta 
    """
    uv = np.asarray(uv).reshape(-1,2)
    r_theta = np.zeros_like(uv) 
    r_theta[:,0] = np.apply_along_axis(np.linalg.norm, 1, uv)
    
    """
    NOTE: Since desired angle is different from the angle that arctan2 outputs;
    the uv array is transformed (multiply by -1) and atan2 is called with (u,v)
    Only to ensure we get the angle per the Wind convention
    """
    uv = np.asarray( -1*np.matrix(uv))  # create new uv object
    r_theta[:,1] = (np.rad2deg(np.arctan2(uv[:,0], uv[:,1])) + 360) % 360   # 0 to 360
    #return np.asarray(r_theta)
    return r_theta
