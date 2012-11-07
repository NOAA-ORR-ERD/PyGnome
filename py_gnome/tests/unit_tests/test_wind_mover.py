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

def test_read_file_init():
    """
    initialize from a long wind file
    """
    file = r"SampleData/WindDataFromGnome.WND"
    wm = movers.WindMover(file=file)
    assert True

def test_timeseries():
    """
    initialize from timeseries and update value
    """
    now = time_utils.round_time( datetime.now(), roundTo=1)   # WindMover rounds data to 1 sec
    val = np.zeros((2,), dtype=basic_types.datetime_value_pair)
    val['time'] = np.datetime64(now.isoformat())
    val['time'][1] = val['time'][1].astype(object) + timedelta(hours=3)
    
    val['value']['u'] = (0,100)
    val['value']['v'] = (50,10)
          
    wm  = movers.WindMover(timeseries=val)
    print val
    print "------------"
    print wm.timeseries
    np.testing.assert_equal(wm.timeseries['time'], val['time'], "time provided during initialization does not match the time in WindMover.timeseries")
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
    time_val['value'][0] = (0., 100.)

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
        delta = np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point)
        delta = np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point)
        actual =np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point) 
        for ix in range(0,2):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            print "Time step [sec]: " + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            print "WindMover.time_series C++:" + str(self.wm.timeseries)
            print "WindMover.get_time_value C++:" + str( self.wm.get_time_value(curr_time))
            delta[ix] = self.wm.get_move(self.pSpill, self.time_step, curr_time)
            
            actual[ix] = self._expected_move()
            
        # the results should be independent of model time
        tol = 1e-8
        print "C++ delta-lat: ";   print str(delta['lat'])
        print "Expected delta-lat: "; print str(actual['lat'])
        print "C++ delta-long: ";  print str(delta['long'])
        print "Expected delta-long: ";print str(actual['long'])
        np.testing.assert_allclose(delta['lat'], actual['lat'], tol, tol,
                                  "get_time_value is not within a tolerance of " + str(tol), 0)
        np.testing.assert_allclose(delta['long'], actual['long'], tol, tol,
                              "get_time_value is not within a tolerance of "+str(tol), 0)
       
    def test_update_wind_vel(self):
        self.time_val['value'][0] = (10., 50.)
        self.wm.timeseries = self.time_val   # update time series
        self.test_get_move()
    
    def _expected_move(self):
        """
        Put the expected move logic in separate private method
        """
        # expected move
        print "wind_vel used for check:" + str( self.time_val[0]['value'])
        exp = np.zeros( (self.pSpill.num_LEs, 3) )
        exp[:,0] = self.pSpill['windages']*self.time_val[0]['value']['u']*self.time_step # 'u'
        exp[:,1] = self.pSpill['windages']*self.time_val[0]['value']['v']*self.time_step # 'v'

        xform = projections.FlatEarthProjection.meters_to_latlon(exp, self.pSpill['positions'])

        actual = np.zeros((len(exp),), dtype=basic_types.world_point)
        actual ['lat'] = xform[:, 1]
        actual ['long'] = xform[:, 0]
        
        return actual
        
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
    tw.test_update_wind_vel()