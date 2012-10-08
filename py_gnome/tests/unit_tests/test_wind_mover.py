#!/usr/bin/env python

"""
unit tests for cy_wind_mover wrapper

designed to be run with py.test
"""

import numpy as np
#from math import sin,cos,pi
#from random import random


from gnome import basic_types
from gnome.cy_gnome import cy_wind_mover
from gnome.cy_gnome import cy_ossm_time
from gnome import greenwich

def test_init(): # can we create a wind_mover?
    wm = cy_wind_mover.CyWindMover()
    assert True

class Common():
    """
    test setting up and moving four particles
    
    Base class that initializes stuff that is common for multiple cy_wind_mover objects
    """
    
    #################
    # create arrays #
    #################
    num_le = 4  # test on 4 LEs
    ref  =  np.empty((num_le,), dtype=basic_types.world_point)   # LEs - initial locations
    wind =  np.empty((num_le,), dtype=np.double) # windage
    const_wind = np.empty((1,), dtype=basic_types.velocity_rec) # constant wind
    #uncertain_ra = np.empty((4,), dtype=basic_types.wind_uncertain_rec)    # one uncertain rec per le

    f_sigma_theta = 1  # ?? 
    f_sigma_vel   = 1  # ??
    
    time_step = 1
    
    def __init__(self):
        self.model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds
        ################
        # init. arrays #
        ################
        self.ref[:] = 1.
        self.ref[:]['z'] = 0 # particles will not move via wind if z>0
    
        self.wind[:] = 1
        # Straight south wind... 100! meters per second
        self.const_wind['u'] =  0  # meters per second?
        self.const_wind['v'] = 100 # 

class ConstantWind(Common):
    """
    Wind Mover object instantiated with a constant wind using member method set_constant_wind(...)
    Used for test setup
    """
    wm = cy_wind_mover.CyWindMover()
    
    def __init__(self):
        Common.__init__(self)
        self.delta = np.zeros((self.num_le,), dtype=basic_types.world_point)
        self.wm.set_constant_wind(self.const_wind['u'], self.const_wind['v'])
    
    def test_move(self):
        """ forecast move """
        self.wm.get_move(self.model_time,
                         self.time_step, 
                         self.ref,
                         self.delta,
                         self.wind,
                         )
              
class ConstantWindWithOSSM(Common):
    """
    This defines the OSSMTimeValue_c object using the CyOSSMTime class, then uses the set_ossm method of
    CyWindMover object to set the time_dep member of the underlying WindMover_c
    C++ object
    
    Used for test setup
    """
    wm = cy_wind_mover.CyWindMover()
    
    def __init__(self):
        Common.__init__(self)
        self.delta = np.zeros((self.num_le,), dtype=basic_types.world_point)
        
        time_val = np.empty((1,), dtype=basic_types.time_value_pair)
        time_val['time'] = 0   # should not matter
        time_val['value']= self.const_wind
        self.ossm = cy_ossm_time.CyOSSMTime(timeseries=time_val,
                                        units=basic_types.velocity_units.knots)
        self.wm.set_ossm(self.ossm)
        
    def test_move(self):
        self.wm.get_move(self.model_time,
                         self.time_step, 
                         self.ref,
                         self.delta,
                         self.wind,
                         )
        
        

def test_constant_wind():
    """
    The result of get_move should be the same irrespective of whether we use OSSM time object
    or the fConstantValue member of the CyWindMover object
    Use the setup in ConstantWind and ConstantWindWithOSSM for this test   
    """
    cw = ConstantWind()
    cww_ossm = ConstantWindWithOSSM()
    
    # the move should be the same from both objects
    cw.test_move()
    cww_ossm.test_move()
    print cw.delta
    print cww_ossm.delta
    np.testing.assert_equal(cw.delta, cww_ossm.delta, "test_constant_wind() failed", 0)
    
    
class TestVariableWind():
    """
    Uses OSSMTimeValue_c to define a variable wind - variable wind has 'v' component, so movement
    should only be in 'lat' direction of world point
    
    Leave as a class as we may add more methods to it for testing 
    """
    wm = cy_wind_mover.CyWindMover()
    cm = Common()
    delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)
    
    time_val = np.zeros((2,), dtype=basic_types.time_value_pair)
    time_val['time'][:] = np.add([0,3600], cm.model_time)   # change after 1 hour
    time_val['value']['v'][:] = [100,200]
    
    # CyOSSMTime needs the same scope as CyWindMover because CyWindMover uses the C++
    # pointer defined in CyOSSMTime.time_dep. This must be defined for the scope
    # of CyWindMover 
    ossm = cy_ossm_time.CyOSSMTime(timeseries=time_val,
                                    units=basic_types.velocity_units.knots)
    wm.set_ossm(ossm)
        
    def test_move(self):
        for x in range(0,3):
            vary_time = x*1800
            self.wm.get_move(self.cm.model_time + vary_time,
                             self.cm.time_step, 
                             self.cm.ref,
                             self.delta,
                             self.cm.wind,
                             )
            print self.delta
            assert np.all(self.delta['lat'] != 0)
            assert np.all(self.delta['long'] == 0)
            assert np.all(self.delta['z'] == 0)