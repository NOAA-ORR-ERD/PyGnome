# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:15:03 2012

@author: brian.zelenke
"""

from __future__ import division
import numpy as np


def reckon_GNOMEstyle(lat,lon,distance,bearing):
    """
    Given a start point, initial bearing, and distance, calculate the
    destination point along a (shortest distance) great circle arc -- using the
    same formula as GNOME.  lat and lon are in decimal degrees. bearing, also
    in decimal degrees, is measured clockwise from north.  distance is in
    meters.
    """
    EarthRadius=6371.*1000. #Earth's mean radius, in m.
    
    #Convert from degrees to radians.
    lat=np.deg2rad(lat)
    lon=np.deg2rad(lon)
    bearing=np.deg2rad(bearing)
    
    #Convert linear distance to angular distance (in radians).
    distance=distance/EarthRadius
    
    latout=np.arcsin(np.sin(lat)*np.cos(distance)+np.cos(lat)*np.sin(distance)*np.cos(bearing))
    lonout=lon+np.arctan2(np.sin(bearing)*np.sin(distance)*np.cos(lat),np.cos(distance)-np.sin(lat)*np.sin(latout))
    
    #Convert from radians to degrees.
    latout=np.rad2deg(latout)
    lonout=np.rad2deg(lonout)
    
    return latout,lonout
