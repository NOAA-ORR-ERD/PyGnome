# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:38:03 PST 2012
Modified on Mon May 1 15:30:00 PST 2012

@author: brian.zelenke
You should  be able to `git checkout prototype; cd py_gnome; python setup.py develop` and then run this script.
"""

from __future__ import division
from gnome import c_gnome
from gnome.basic_types import le_rec, oil_status
import numpy as np
import geodesic_calcs

class spill:
    def __init__(self,npra,uncertain):
        '''
        Create a dictionary of the attributes used to initalize the model.
        "npra" is the numpy array filled with the spill(s) parameters.
        "uncertain" is the uncertainty value to use of each spill (can be set
        to "False" to turn off uncertanity calculation.)
        '''
        self.npra=npra
        self.uncertain=uncertain


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
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the north over a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=0.0 #U component of wind velocity in m/s.
    WindV=1.0 #V component of wind velocity in m/s.
    Tstep=1000.0 #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.    
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due north.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*Tstep #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.



def test_southward():
    '''
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the south over a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=0.0 #U component of wind velocity in m/s.
    WindV=-1.0 #V component of wind velocity in m/s.
    Tstep=1000.0 #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.    
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due south.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*Tstep #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.



def test_eastward():
    '''
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the east over a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=1.0 #U component of wind velocity in m/s.
    WindV=0.0 #V component of wind velocity in m/s.
    Tstep=1000.0 #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due east.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*Tstep #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.



def test_westward():
    '''
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the north over a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=-1.0 #U component of wind velocity in m/s.
    WindV=0.0 #V component of wind velocity in m/s.
    Tstep=1000.0 #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due west.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*Tstep #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.



def test_northeastward():
    '''
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the north over a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=0.70710679 #U component of wind velocity in m/s.
    WindV=0.70710679 #V component of wind velocity in m/s.
    Tstep=1000.0 #Run duration (time-step), in seconds.
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    print npra #See what the numpy array being input into the get_move function looks like.
    m.get_move(Tstep,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex
    print npra #See what the numpy array looks like after processing by the get_move function.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km northeast.  Compare the end
    #   coordinate returned by pyGNOME with the analytical solution.
    #==========================================================================
    
    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*Tstep #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    if np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon):
        print "pyGNOME latitude and longitude (%g, %g) are equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    else:
        print "Error:  pyGNOME latitude and longitude (%g, %g) are NOT equivalent to analytical latitude and longitude (%g, %g)." %(EndLat,EndLon,knownLat,knownLon)
    
    assert(np.allclose(EndLat,knownLat) and np.allclose(EndLon,knownLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.


def test_northward_stepwise():
    '''
    Use the Python implementation of GNOME to test moving a particle, forced
    only by wind, to the north over multiple time-steps.  Ensure that the end
    coordinate returned when subdividing a run into multiple time-steps is the
    same as calculated when moving the particle the same distance/direction in
    a single time-step.
    '''
    
    #Define input parameters.
    num_particles=1.0 #Number of LEs
    LatIn=26.0 #Starting latitude in decimal degrees.
    LonIn=-88.0 #Starting longitude in decimal degrees.
    WindPrct=1.0 #Windage in percentage (as a decimal).
    WindU=0.0 #U component of wind velocity in m/s.
    WindV=1.0 #V component of wind velocity in m/s.
    TotalT=1000.0 #Total run duration, in seconds.
    Tdivisor=10.0 #Number to divide TotalT by into time-steps (typically an integer evenly divides TotalT).
    
    #Fill-in array with the input parameters.
    npra = np.zeros(num_particles, dtype=le_rec) #Initialize and empty array of zeros.
    npra["p"]=(LonIn,LatIn) #World point longitude,latitude.
    npra["windage"]=(WindPrct) #Windage in percentage (as a decimal.)
    npra["status_code"]=oil_status.status_in_water #Specify the particle state.
    
    #Initalize the model.
    SpillDict=spill(npra,False) #Set uncertanity to "False" to not use uncertanity.
    Spill=[SpillDict] #Make a list out of the dictionary.
    c_gnome.initialize_model(Spill) #Pass in the list.
    m = c_gnome.wind_mover((WindU,WindV)) #U,V velocity component in m/s.
    
    
    #==========================================================================
    #   Calculate the end coordinate analytically from the input parameters.
    #==========================================================================

    #Convert from Cartesian (U&V) to polar (magnitude and angle).
    dist=np.hypot(WindU,WindV)*TotalT #(m/s)*s=distance in meters.
    azimuth=np.mod(np.rad2deg(np.arctan2(WindU,WindV)),360.0) #Degrees clockwise from north.
    
    knownLat,knownLon=geodesic_calcs.reckon_GNOMEstyle(LatIn,LonIn,dist,azimuth) #return EndLat,EndLon
    
    #==========================================================================
    #   Calculate the end coordinate in a single time-step via pyGNOME.
    #==========================================================================

    m.get_move(TotalT,npra,False,0) #Timestep(seconds),NumpyArray,Uncertanity,SpillIndex.
    
    #Ending coordinate.
    EndLat=npra["p"]["p_lat"][0]
    EndLon=npra["p"]["p_long"][0]
    
    #==========================================================================
    #   Calculate the end coordinate over a series of time-steps via pyGNOME.
    #==========================================================================
    
    npra["p"]=(LonIn,LatIn) #Set the point field of the numpy array back to the start coordinate.
    Tsteps=np.tile(TotalT/Tdivisor,(Tdivisor,1)) #Create a vector of time-steps (in seconds) that evenly spans total run duration.
    LatSteps=np.tile(np.nan,Tsteps.shape) #Preallocate vectors for subsequent loop.
    LonSteps=np.tile(np.nan,Tsteps.shape)
    
    idx=-1
    for iStep in Tsteps:
        m.get_move(iStep,npra,False,0)
        idx=idx+1
        LatSteps[idx]=npra["p"]["p_lat"][0]
        LonSteps[idx]=npra["p"]["p_long"][0]
    
    
    
    #==========================================================================
    #   Input parameters defined above specify that the given particle be
    #   moved [100%]*[1 m/s]*[1000 s] = 1 km due north.  Compare the end
    #   coordinate reached by having pyGNOME divide this trajectory into a
    #   series of (time) steps with the analytical solution.
    #==========================================================================
    
    
    
    if np.allclose(LatSteps[-1],knownLat) and np.allclose(LonSteps[-1],knownLon) and np.allclose(LatSteps[-1],EndLat) and np.allclose(LonSteps[-1],EndLon):
        print "pyGNOME latitude & longitude (%g, %g) at the end of %g time-steps are equivalent to both analytical latitude & longitude (%g, %g) and pyGNOME latitude & longitude (%g, %g) calculated with a single time-step." %(LatSteps[-1],LonSteps[-1],Tdivisor,knownLat,knownLon,EndLat,EndLon)
    else:
        print "Error:  pyGNOME latitude & longitude (%g, %g) at the end of %g time-steps are NOT equivalent to both analytical latitude & longitude (%g, %g) and pyGNOME latitude & longitude (%g, %g) calculated with a single time-step." %(LatSteps[-1],LonSteps[-1],Tdivisor,knownLat,knownLon,EndLat,EndLon)
    
    assert(np.allclose(LatSteps[-1],knownLat) and np.allclose(LonSteps[-1],knownLon) and np.allclose(LatSteps[-1],EndLat) and np.allclose(LonSteps[-1],EndLon)) #If the analytic and pyGNOME results differ too much, thow an error so the test fails.
