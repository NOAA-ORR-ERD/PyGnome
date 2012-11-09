from gnome import movers

from gnome import basic_types, spill
from gnome.utilities import time_utils 
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
        movers.WindMover(timeseries=wind_vel)

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

def test_timeseries():
    """
    initialize from timeseries and update value
    """
    now = time_utils.round_time( datetime.now(), roundTo=1)   # WindMover rounds data to 1 sec
    val = np.zeros((5,), dtype=basic_types.datetime_value_pair)
    val['time'] = [datetime(2012,11,06,20,10,i) for i in range(5)]
    
    val['value']['u'] = [i for i in range(0,10,2)]
    val['value']['v'] = [i for i in range(10,20,2)]
          
    wm  = movers.WindMover(timeseries=val)
    _defaults(wm)   # also check defaults
    
    print time_utils.round_time(val, roundTo=1)
    print "------------"
    print time_utils.round_time(wm.timeseries['time'], roundTo=1)
    np.testing.assert_equal(time_utils.round_time(wm.timeseries['time'], roundTo=1), 
                           time_utils.round_time(val['time'], roundTo=1), "time provided during initialization does not match the time in WindMover.timeseries")
    np.testing.assert_equal(wm.timeseries['value'], val['value'], "velocity_rec returned by WindMover.timeseries is not the same as what was input during initialization")
    

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

    time_val = np.zeros((1,), dtype=basic_types.datetime_value_pair)
    time_val['time'][0] = np.datetime64( rel_time.isoformat() )
    time_val['value'][0] = (5., 100.)

    wm = movers.WindMover(timeseries=time_val)

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

        # make sure clean up is happening fine
        #delta = np.zeros((self.pSpill.num_LEs,3), dtype=basic_types.world_point)
        #actual =np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point) 
        for ix in range(1):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            delta = self.wm.get_move(self.pSpill, self.time_step, curr_time)
            actual = self._expected_move()
            
            # the results should be independent of model time
            tol = 1e-8
            np.testing.assert_allclose(delta, actual, tol, tol,
                                       "WindMover.get_move() is not within a tolerance of " + str(tol), 0)
            
            print "Time step [sec]: \t\t" + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            print "WindMover.get_time_value C++:\t" + str( self.wm.get_time_value(curr_time))
            print "wind_vel for _expected_move:\t" + str( self.time_val[0]['value'])
            print "C++ delta-long: " + "\t\t" +  str(delta[0])
            print "Expected delta-long: "+ "\t\t" + str(actual[0])
            print "C++ delta-lat  : " + "\t\t" + str(delta[1])
            print "Expected delta-lat: "+ "\t\t" + str(actual[1])
                   
    def test_update_wind_vel(self):
        self.time_val['value'][0] = (10., 50.)
        self.wm.timeseries = self.time_val   # update time series
        self.test_get_move()
    
    def _expected_move(self):
        """
        Put the expected move logic in separate private method
        """
        # expected move
        exp = np.zeros( (self.pSpill.num_LEs, 3) )
        exp[:,0] = self.pSpill['windages']*self.time_val[0]['value']['u']*self.time_step # 'u'
        exp[:,1] = self.pSpill['windages']*self.time_val[0]['value']['v']*self.time_step # 'v'

        xform = projections.FlatEarthProjection.meters_to_lonlat(exp, self.pSpill['positions'])
        
        return xform
        
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
    tw.test_update_wind_vel()