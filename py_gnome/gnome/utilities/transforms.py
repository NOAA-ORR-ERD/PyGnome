import numpy as np
import math

def r_theta_to_uv_wind(r_theta):
    """
    Applies the r_theta_to_xy for the transformation - for this 0deg corresponds with (x,y) = (1,0)
    Since the wind tranformation is rotated from this by 90deg, such that:
        (u,v) = (0,1) corresponds with 0deg, while (x,y) = (1,0) corresponds with 0 deg.
    Therefore, u = -y and v = x after the transformation
    
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding xy cartesian velocity vector
    """
    xy = r_theta_to_xy(r_theta)
    
    # following gives -0, just looks odd - use matrix multiply instead!
    #uv[:,0] = -1*xy[:,1]
    #uv[:,1] = xy[:,0]
    
    xform = np.matrix([[0,1],[-1,0]])
    return np.array( np.dot(xy, xform))

def wind_uv_to_r_theta(uv):
    """
    Applies the transformation to get wind_uv in xy format. The xy axis is rotated 90 deg from uv.
        [x] = [0, -1] [u]
        [y] = [1,  0] [v]
     
    then apply xy_to_r_theta transformation. For (x,y) 0deg corresponds with (x,y) = (1,0)
    The wind tranformation is rotated from this by 90deg, such that:
        (u,v) = (0,1) corresponds with 0deg, while (x,y) = (1,0) corresponds with 0 deg.
    
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding xy cartesian velocity vector
    """
    # apply the inverse of the xform in r_theta_to_uv_wind
    xform = np.matrix([[0,-1],[1,0]])
    xy = np.array( np.dot(uv, xform))
    return xy_to_r_theta(xy)

def r_theta_to_xy(r_theta):
    """
    This applies the following transformation:
        x = magnitude * cos(theta)
        y = magnitude * sin(theta)
         
    This is just a polar to cartesian transform where 0deg corresponds with (x,y) = (1,0)
    and the angle increases to 360deg counter-clockwise around the circle 
    
    :param r_theta: NX2 numpy array containing r = r_theta[:,0], theta = r_theta[:,1]. 
                    Theta is in degrees between 0 and 360.
    :returns: NX2 numpy array containing the corresponding xy cartesian velocity vector
    """
    r_theta = np.asarray(r_theta).reshape(-1,2)
    if np.any(r_theta[:,1] > 360) or np.any(r_theta[:,1] < 0):
        raise ValueError("input angle in r_theta[:,1] must be between 0 and 360")
    
    if np.any(r_theta[:,0] <= 0):
        raise ValueError("input magnitude in r_theta[:,0] must be greater than 0")
    
    theta = np.deg2rad(r_theta[:,1])
    xy = np.zeros_like(r_theta)
    # 15 digits appears to be the precision of float64
    xy[:,0] = np.round( r_theta[:,0]*np.cos(theta), decimals=14)
    xy[:,1] = np.round( r_theta[:,0]*np.sin(theta), decimals=14)
    return xy

def xy_to_r_theta(xy):
    """
    This applies the following transformation:
        r = sqrt( u**2 + v**2)
        y = arctan2( y / x) 
         
    This is just a polar to cartesian transform where 0deg corresponds with (x,y) = (1,0)
    and the angle increases to 360deg counter-clockwise around the circle 
    
    :param xy: NX2 numpy array, where each row corresponds with a velocity vector 
    :returns: NX2 numpy array containing polar coordinates r_theta 
    """
    xy = np.asarray(xy).reshape(-1,2)
    r_theta = np.zeros_like(xy) 
    r_theta[:,0] = np.apply_along_axis(np.linalg.norm, 1, xy)
    r_theta[:,1] = (np.rad2deg(np.arctan2(xy[:,1], xy[:,0])) + 360) % 360   # 0 to 360
    return r_theta