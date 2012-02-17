# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:38:03 2012

@author: brian.zelenke
"""

from __future__ import division
from gnome import c_gnome
from gnome.basic_types import le_rec, status_in_water
import numpy as np
import geodesic_calcs


def test_init():
    '''
    Test that the wind_mover function runs without crashing.
    '''
    m = c_gnome.wind_mover((0,1)) #U,V velocity component in m/s.
    print m
    assert False #Force this test to fail so that NOSE will print output to the command window.
    return m
    
    
def test_northward():
    '''
    Use the Python implementation of GNOME to test moving a particle to the
    north (at present, forced via wind only).
    '''
    
    #Define input parameters.
    num_particles=1 #Number of LEs
    LatIn=26. #Starting latitude in decimal degrees.
    LonIn=-88. #Starting longitude in decimal degrees.
    WindPrct=1. #Windage in percentage (as a decimal).
    WindU=0. #U component of wind velocity in m/s.
    WindV=1. #V component of wind velocity in m/s.
    Tstep=1000. #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=status_in_water #Specify the particle state.
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra) #Timestep(seconds),NumpyArray.
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due north.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,WindV*Tstep,0.)
    #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.
    