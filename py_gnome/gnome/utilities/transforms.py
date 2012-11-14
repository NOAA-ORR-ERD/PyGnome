import numpy as np
import math

def r_theta_to_uv(r_theta):
    """
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding uv cartesian velocity vector
    """
    r_theta = np.asarray(r_theta).reshape(-1,2)
    if np.any(r_theta[:,1] > 360) or np.any(r_theta[:,1] < 0):
        raise ValueError("input angle in r_theta[:,1] must be between 0 and 360")
    
    if np.any(r_theta[:,0] <= 0):
        raise ValueError("input magnitude in r_theta[:,0] must be greater than 0")
    
    theta = np.deg2rad(r_theta[:,1])
    uv = np.zeros_like(r_theta)
    # 15 digits appears to be the precision of float64
    uv[:,0] = np.round( r_theta[:,0]*np.cos(theta), decimals=14)
    uv[:,1] = np.round( r_theta[:,0]*np.sin(theta), decimals=14)
    return uv

def uv_to_r_theta(uv):
    """
    :param uv: NX2 numpy array, where each row corresponds with a velocity vector 
    :returns: NX2 numpy array containing polar coordinates r_theta 
    """
    uv = np.asarray(uv).reshape(-1,2)
    r_theta = np.zeros_like(uv) 
    r_theta[:,0] = np.apply_along_axis(np.linalg.norm, 1, uv)
    r_theta[:,1] = (np.rad2deg(np.arctan2(uv[:,1], uv[:,0])) + 360) % 360   # 0 to 360
    return r_theta