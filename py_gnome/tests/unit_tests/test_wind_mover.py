from gnome import movers

from gnome import basic_types, spill
from gnome.utilities import time_utils, transforms 
from gnome import greenwich
from gnome.utilities import projections

import numpy as np

from datetime import timedelta, datetime
import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with pytest.raises(ValueError):
        movers.WindMover()
        wind_vel = np.zeros((1,), basic_types.velocity_rec)
        movers.WindMover(timeseries=wind_vel, data_format=basic_types.data_format.wind_uv)

def test_init():
    """
    test setting the properties of the object
    """
    file = r"SampleData/WindDataFromGnome.WND"
    wm = movers.WindMover(file=file, uncertain_duration=1, is_active=False,
                 uncertain_time_delay=2, uncertain_speed_scale=3, uncertain_angle_scale=4)
    assert wm.is_active == False
    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4


def test_properties():
    """
    test setting the properties of the object
    """
    file = r"SampleData/WindDataFromGnome.WND"
    wm = movers.WindMover(file=file)
    
    wm.uncertain_duration = 1
    wm.uncertain_time_delay = 2
    wm.uncertain_speed_scale = 3
    wm.uncertain_angle_scale = 4
    
    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4

def _defaults(wm):
    """
    checks the default properties of the WindMover object as given in the input are as expected
    """
    assert wm.is_active == True
    assert wm.uncertain_duration == 10800
    assert wm.uncertain_time_delay == 0
    assert wm.uncertain_speed_scale == 2
    assert wm.uncertain_angle_scale == 0.4

def test_read_file_init():
    """
    initialize from a long wind file
    """
    file = r"SampleData/WindDataFromGnome.WND"
    wm = movers.WindMover(file=file)
    _defaults(wm)   # check defaults set correctly
    assert True


now = time_utils.round_time( datetime.now(), roundTo=1)   # WindMover rounds data to 1 sec
val = np.zeros((5,), dtype=basic_types.datetime_value_2d)
val['time'] = [datetime(2012,11,06,20,10,i) for i in range(5)]

val['value'][:,0] = [i for i in range(1,10,2)]
val['value'][:,1] = [i for i in range(10,20,2)]

def test_timeseries_r_theta():
    """
    init using (r,theta)
    Default data_format=basic_types.data_format.magnitude_direction
    """
    wm  = movers.WindMover(timeseries=val,data_format=basic_types.data_format.magnitude_direction)
    np.testing.assert_equal(time_utils.round_time(wm.timeseries['time'], roundTo=1), val['time'],  
                            "time provided during initialization does not match the time in WindMover.timeseries", 0)
    np.testing.assert_allclose(wm.timeseries['value'], val['value'], 1e-10, 1e-10, 
                               "velocity_rec returned by WindMover.timeseries is not the same as what was input during initialization", 0)
    

          
def test_get_time_value():
    """
    Initialize from timeseries and test the get_time_value method 
    """
    wm  = movers.WindMover(timeseries=val,data_format=basic_types.data_format.wind_uv)
    _defaults(wm)   # also check defaults
    
    # check get_time_value()
    gtime_val = wm.get_time_value(val['time'])
    np.all(gtime_val == val['value'])
    
    
def test_timeseries_uv():
    """
    initialize from timeseries and update value
    """
    wm  = movers.WindMover(timeseries=val,data_format=basic_types.data_format.wind_uv)
    _defaults(wm)   # also check defaults
    
    # Test timeseries ..
    print "------------"
    print time_utils.round_time(val, roundTo=1)
    print time_utils.round_time(wm.timeseries['time'], roundTo=1)
    print "------------"
    print transforms.uv_to_r_theta_wind(val['value'])
    print wm.timeseries['value']
    print "------------"
    np.testing.assert_equal(time_utils.round_time(wm.timeseries['time'], roundTo=1), 
                          time_utils.round_time(val['time'], roundTo=1), "time provided during initialization does not match the time in WindMover.timeseries")
    np.testing.assert_equal(wm.timeseries['value'], transforms.uv_to_r_theta_wind(val['value']), 
                            "velocity_rec returned by WindMover.timeseries is not the same as what was input during initialization")
    

class TestWindMover():
    """
    gnome.WindMover() test

    TODO: Move it to separate file
    """
    num_le = 5
    start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
    start_pos += (3.,6.,0.)
    rel_time = datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
    time_step = 15*60 # seconds

    pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time, persist=-1)

    time_val = np.zeros((1,), dtype=basic_types.datetime_value_2d)  # value is given as (r,theta)
    time_val['time']  = np.datetime64( rel_time.isoformat() )
    time_val['value'] = (5., 0.)

    wm = movers.WindMover(timeseries=time_val,data_format=basic_types.data_format.magnitude_direction)

    def test_string_representation_matches_repr_method(self):
        assert repr(self.wm) == 'Wind Mover'
        assert str(self.wm) == 'Wind Mover'

    def test_id_matches_builtin_id(self):
        assert id(self.wm) == self.wm.id

    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """
        self.pSpill.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.prepare_for_model_step(self.model_time, self.time_step)

        for ix in range(2):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            delta = self.wm.get_move(self.pSpill, self.time_step, curr_time)
            actual = self._expected_move()
            
            # the results should be independent of model time
            tol = 1e-8
            np.testing.assert_allclose(delta, actual, tol, tol,
                                       "WindMover.get_move() is not within a tolerance of " + str(tol), 0)
            
            print "Time step [sec]: \t\t" + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            print "WindMover.get_time_value C++:\t" + str( self.wm.get_time_value(curr_time))
            #print "wind_vel for _expected_move:\t" + str( self.time_val['value'])
            print "C++ delta-: " ; print str(delta)
            print "Expected delta: "; print str(actual)
                   
    # TODO: Currently the values are only given in (r,theta) format
    def test_update_wind_vel(self):
        self.time_val['value'] = (1., 120.) # again given as (r, theta)
        self.wm.timeseries = self.time_val  # update time series
        self.test_get_move()
    
    def _expected_move(self):
        """
        Put the expected move logic in separate private method
        """
        # expected move
        uv = transforms.r_theta_to_uv_wind(self.time_val['value'])
        exp = np.zeros( (self.pSpill.num_LEs, 3) )
        exp[:,0] = self.pSpill['windages']*uv[0,0]*self.time_step # 'u'
        exp[:,1] = self.pSpill['windages']*uv[0,1]*self.time_step # 'v'

        xform = projections.FlatEarthProjection.meters_to_lonlat(exp, self.pSpill['positions'])
        
        return xform
        
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
    tw.test_update_wind_vel()