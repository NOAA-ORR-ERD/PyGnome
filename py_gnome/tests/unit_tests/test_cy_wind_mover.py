#!/usr/bin/env python

"""
unit tests for cy_wind_mover wrapper

designed to be run with py.test
"""

import numpy as np
import datetime

from gnome import basic_types
from gnome.cy_gnome import cy_wind_mover
from gnome.cy_gnome import cy_ossm_time
from gnome.utilities import time_utils 

from gnome.utilities import projections


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
    ref  =  np.zeros((num_le,), dtype=basic_types.world_point)   # LEs - initial locations
    wind =  np.zeros((num_le,), dtype=np.double) # windage
    status = np.empty((num_le,), dtype=basic_types.status_code_type)
    const_wind = np.zeros((1,), dtype=basic_types.velocity_rec) # constant wind
    setSizes = np.zeros((1,), dtype=np.int) # number of LEs in 1 uncertainty spill - simple test
    
    time_step = 60
    
    def __init__(self):
        time = datetime.datetime(2012, 8, 20, 13)
        self.model_time = time_utils.date_to_sec( time)
        ################
        # init. arrays #
        ################
        self.ref[:] = 1.
        self.ref[:]['z'] = 0 # particles will not move via wind if z>0
    
        self.wind[:] = [1./100.,2./100.,3./100.,4./100.]
        # Straight south wind... 100! meters per second
        self.const_wind['u'] = 50  # meters per second?
        self.const_wind['v'] = 100 #
        self.status[:] = basic_types.oil_status.in_water
        self.setSizes[0] = self.num_le 

class ConstantWind(Common):
    """
    Wind Mover object instantiated with a constant wind using member method set_constant_wind(...)
    Used for test setup
    """
    wm = cy_wind_mover.CyWindMover()
    
    def __init__(self):
        Common.__init__(self)
        self.delta = np.zeros((self.num_le,), dtype=basic_types.world_point)
        self.u_delta = np.zeros((self.num_le,), dtype=basic_types.world_point)
        self.wm.set_constant_wind(self.const_wind['u'], self.const_wind['v'])
    
    def test_move(self):
        """ forecast move """
        self.wm.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.get_move(self.model_time,
                         self.time_step, 
                         self.ref,
                         self.delta,
                         self.wind,
                         self.status,
                         basic_types.spill_type.forecast,
                         0)
        
    def test_move_uncertain(self):
       """ uncertain LEs """
       self.wm.prepare_for_model_step(self.model_time, self.time_step, len(self.setSizes), self.setSizes)
       self.wm.get_move(self.model_time,
                        self.time_step, 
                        self.ref,
                        self.u_delta,
                        self.wind,
                        self.status,
                        basic_types.spill_type.forecast,
                        0)
        
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
        self.u_delta = np.zeros((self.num_le,), dtype=basic_types.world_point)
        
        time_val = np.empty((1,), dtype=basic_types.time_value_pair)
        time_val['time'] = 0   # should not matter
        time_val['value']= self.const_wind
        self.ossm = cy_ossm_time.CyOSSMTime(timeseries=time_val)
        self.wm.set_ossm(self.ossm)
        
    def test_move(self):

        self.wm.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.get_move(self.model_time,
                         self.time_step, 
                         self.ref,
                         self.delta,
                         self.wind,
                         self.status,
                         basic_types.spill_type.forecast,
                         0)
        
    def test_move_uncertain(self):
       """ uncertain LEs """
       self.wm.prepare_for_model_step(self.model_time, self.time_step, len(self.setSizes), self.setSizes)
       self.wm.get_move(self.model_time,
                        self.time_step, 
                        self.ref,
                        self.u_delta,
                        self.wind,
                        self.status,
                        basic_types.spill_type.forecast,
                        0)
        
class TestConstantWind():
    cw = ConstantWind()
    cw.test_move()
    
    cww_ossm = ConstantWindWithOSSM()
    
    cww_ossm.test_move()
    
    def test_constant_wind(self):
        """
        The result of get_move should be the same irrespective of whether we use OSSM time object
        or the fConstantValue member of the CyWindMover object
        Use the setup in ConstantWind and ConstantWindWithOSSM for this test   
        """
        # the move should be the same from both objects
        print self.cw.delta
        print self.cww_ossm.delta
        np.testing.assert_equal(self.cw.delta, self.cww_ossm.delta, "test_constant_wind() failed", 0)
        
    def test_move_value(self):
        #meters_per_deg_lat = 111120.00024
        #self.cw.wind/meters_per_deg_lat
        delta = np.zeros( (self.cw.num_le, 3) )
        delta[:,0] = self.cw.wind*self.cw.const_wind['u']*self.cw.time_step # 'u'
        delta[:,1] = self.cw.wind*self.cw.const_wind['v']*self.cw.time_step # 'v'
        
        
        ref = self.cw.ref.view(dtype=basic_types.world_point_type).reshape((-1,3))
        xform = projections.FlatEarthProjection.meters_to_latlon( delta, ref)
        
        actual = np.zeros((self.cw.num_le,), dtype=basic_types.world_point)
        actual ['lat'] = xform[:,1]
        actual ['long'] = xform[:,0]
        
        tol = 1e-10
        np.testing.assert_allclose(self.cw.delta['lat'], actual['lat'], tol, tol, 
                                   "get_time_value is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cw.delta['long'], actual['long'], tol, tol, 
                                   "get_time_value is not within a tolerance of "+str(tol), 0)
    
    def test_move_uncertain(self):
       self.cw.test_move_uncertain()
       self.cww_ossm.test_move_uncertain()
       print "=================================================="
       print " Check move for uncertain LEs (test_move_uncertain)  "
       print "--- ConstandWind ------"
       print "Forecast LEs delta: " 
       print self.cw.delta
       print "Uncertain LEs delta: "
       print self.cw.u_delta
       print "--- ConstandWind with OSSM ------"
       print "Forecast LEs delta: "
       print self.cww_ossm.delta
       print "Uncertain LEs delta: "
       print self.cww_ossm.u_delta
       assert np.all(self.cw.delta != self.cw.u_delta)
       assert np.all(self.cww_ossm.delta != self.cww_ossm.u_delta)
    
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
    ossm = cy_ossm_time.CyOSSMTime(timeseries=time_val)
    wm.set_ossm(ossm)
        
    def test_move(self):
        for x in range(0,3):
            vary_time = x*1800
            self.wm.prepare_for_model_step(self.cm.model_time + vary_time, self.cm.time_step)
            self.wm.get_move(self.cm.model_time + vary_time,
                             self.cm.time_step, 
                             self.cm.ref,
                             self.delta,
                             self.cm.wind,
                             self.cm.status,
                             basic_types.spill_type.forecast,
                             0)
            print self.delta
            assert np.all(self.delta['lat'] != 0)
            assert np.all(self.delta['long'] == 0)
            assert np.all(self.delta['z'] == 0)

def test_LE_not_in_water():
    """
    Tests get_move returns 0 for LE's that have a status different from in_water
    """
    wm = cy_wind_mover.CyWindMover()
    cm = Common()
    delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)
    cm.status[:] = 0
    wm.prepare_for_model_step(cm.model_time, cm.time_step)
    wm.get_move(cm.model_time,
                cm.time_step, 
                cm.ref,
                delta,
                cm.wind,
                cm.status,
                basic_types.spill_type.forecast,
                0)
    assert np.all(delta['lat'] == 0)
    assert np.all(delta['long'] == 0)
    assert np.all(delta['z'] == 0)
    
def test_z_greater_than_0():
    """
    If z > 0, then the particle is below the surface and the wind does not act on it. As such, the get_move
    should return 0 for delta
    """
    cw = ConstantWind()
    cw.ref['z'][:2] = 2 # particles 0,1 are not on the surface    
    delta = np.zeros((cw.num_le,), dtype=basic_types.world_point)
    
    cw.test_move()
    
    assert np.all(cw.delta['lat'][0:2] == 0)
    assert np.all(cw.delta['long'][0:2] == 0)
    assert np.all(cw.delta['z'][0:2] == 0)
    
    # for particles in water, there is a non zero delta
    assert np.all(cw.delta['lat'][2:] != 0)
    assert np.all(cw.delta['long'][2:] != 0)
    assert np.all(cw.delta['z'][2:] == 0)
    
    
if __name__=="__main__":
    cw= TestConstantWind()
    cw.test_constant_wind()
    cw.test_move_value()
    cw.test_move_uncertain()
