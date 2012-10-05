#!/usr/bin/env python

"""
unit tests for the wind_mover wrapper

designed to be run with py.test
"""

import numpy as np
from math import sin,cos,pi
from random import random


from gnome import basic_types
from gnome.cy_gnome import cy_wind_mover
from gnome.cy_gnome import cy_ossm_time
from gnome import greenwich

def test_init(): # can we create a wind_mover?
    wm = cy_wind_mover.Cy_wind_mover()
    assert True

class Test_setup_1():
    """
    test setting up and moving four particles
    
    setup is done is the class constructor
    """
    wm = cy_wind_mover.Cy_wind_mover()
    
    model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds


    #################
    # create arrays #
    #################
    wp_ra   =  np.empty((4,), dtype=basic_types.world_point)
    ref_ra  =  np.empty((4,), dtype=basic_types.world_point)
    wind_ra =  np.empty((4,), dtype=np.double)
    time_vals = np.empty((1,), dtype=basic_types.time_value_pair)
    uncertain_ra = np.empty((4,), dtype=basic_types.wind_uncertain_rec)	# one uncertain rec per le

    f_sigma_theta = 1  # ?? 
    f_sigma_vel   = 1  # ??

    ################
    # init. arrays #
    ################

    N = len(wp_ra)
    M = len(time_vals)

    wp_ra[:] = 1.
    ref_ra[:] = 1.

    ref_ra[:]['lat'] *= 1000000 #huh? I thought we were getting rid of the 1e6 stuff.
    ref_ra[:]['long'] *= 1000000 
    ref_ra[:]['z'] = 0 # particles will not move via wind if z>0

    wind_ra[:] = 1
    # Straight south wind... 100! meters per second
    time_vals['value']['u'] =  0  # meters per second?
    time_vals['value']['v'] = 100 # 

    #file = r"SampleData/ConstantWind.WND"
    #ossmT = cy_ossm_time.Cy_ossm_time()
    #err = ossmT.ReadTimeValues(file); # assume default format and units
    wm.set_constantWind(time_vals['value']['u'],time_vals['value']['v'])
   # initialize uncertainty array:
    for x in range(0, N):
        theta = random()*2*pi
        uncertain_ra[x]['randCos'] = cos(theta)
        uncertain_ra[x]['randSin'] = sin(theta)
        
        # ein uncertain array

    ##################
    # call and check #
    ##################



    def test_move(self):
        """ forecast move """

        print '###################'
        print '# init. positions #'
        print '###################'

        print self.wp_ra
        print
        for x in range(0, 1):
            self.wm.get_move(self.model_time,
                             10, # time step -- 10 seconds
                             self.ref_ra,
                             self.wp_ra,
                             self.wind_ra,
                             )
        print self.wp_ra
        
        ## this doesn't test anything, except that it it can run.
        assert True

#    print
#    print '#################'
#    print '# uncertainmove #'
#    print '#################'
#
#    wp_ra[:] = 1.
#    ref_ra[:] = 1.
#
#    ref_ra[:]['p']['p_lat'] *= 1000000
#    ref_ra[:]['p']['p_long'] *= 1000000
#
#    for x in range(0, 1):
#        wm.get_move_uncertain(N, model_time, 10, ref_ra, wp_ra, wind_ra, f_sigma_vel, f_sigma_theta, uncertain_ra)
#
#    print wp_ra
