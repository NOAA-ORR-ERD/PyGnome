from gnome import movers

from gnome import basic_types, spill, weather
from gnome.utilities import time_utils, transforms 
from gnome.utilities import projections

import numpy as np

from datetime import timedelta, datetime
import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with pytest.raises(TypeError):
        movers.WindMover()

# tolerance for np.allclose(..) function
atol = 1e-14
rtol = 1e-14

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
    wind = weather.Wind(file=file)
    wm = movers.WindMover(wind)
    wind_ts = wind.get_timeseries(data_format=basic_types.data_format.wind_uv)
    _defaults(wm)   # check defaults set correctly
    print
    print "====================="
    print "WindMover timeseries [time], [u, v]: "
    print wm.timeseries['time']
    print wm.timeseries['value']
    print "---------------------"
    print "Wind timeseries [time], [u, v]: "
    print wind_ts['time']
    print wind_ts['value']
    np.all(wm.timeseries['time'] == wind_ts['time'])
    assert np.allclose(wm.timeseries['value'], wind_ts['value'], atol, rtol)

@pytest.fixture(scope="module")
def wind(wind_circ):
    """
    Create Wind object using the time series given by the test fixture
    'wind_circ' and the default settings
    """
    dtv_rq = np.zeros((len(wind_circ['rq']),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(2012,11,06,20,10+i,30) for i in range(len(dtv_rq))]
    dtv_rq.value = wind_circ['rq']
    dtv_uv = np.zeros((len(dtv_rq),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.value= wind_circ['uv']
    
    wind  = weather.Wind(timeseries=dtv_rq,data_format=basic_types.data_format.magnitude_direction)
    return {'wind':wind, 'rq': dtv_rq, 'uv': dtv_uv} 
    
def test_timeseries_init(wind):
    """
    test default properties of the object are initialized correctly
    """
    wm = movers.WindMover(wind['wind'])
    _defaults(wm)
    np.all(wm.timeseries['time'] == wind['uv']['time'])
    assert np.allclose(wm.timeseries['value'], wind['uv']['value'], atol, rtol)
    
def test_update_wind(wind):
    """
    create a wind object and update it's timeseries. Make sure the internal C++ WindMover's properties have also changed
    """
    o_wind = wind['wind']           # original wind value
    wm  = movers.WindMover(o_wind)  # define wind mover
    
    t_dtv = np.zeros((3,), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    t_dtv.time = [datetime(2012,11,06,20,0+i,30) for i in range(3)]
    t_dtv.value= np.random.uniform(1,5, (3,2) )
    
    # update wind timeseries - default data_format is magnitude_direction
    o_wind.set_timeseries(t_dtv, data_format=basic_types.data_format.wind_uv) 
    
    np.all(wm.timeseries['time'] == t_dtv.time)
    assert np.allclose(wm.timeseries['value'], t_dtv.value, atol, rtol)
    
    # set the wind timeseries back to test fixture values
    o_wind.set_timeseries(wind['rq'])
    np.all(wm.timeseries['time'] == wind['uv']['time'])
    assert np.allclose(wm.timeseries['value'], wind['uv']['value'], atol, rtol)
   
   
def test_properties(wind):
    """
    test setting the properties of the object
    """
    wm = movers.WindMover(wind['wind'])
    
    wm.uncertain_duration = 1
    wm.uncertain_time_delay = 2
    wm.uncertain_speed_scale = 3
    wm.uncertain_angle_scale = 4
    
    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4    
   

"""
Defined as a function for standard point release spill for testing movers. The way it is used
below, a function here is not required - it is only defined and used here so that 
if this spill is used by multiple test modules, it can simply be moved to conftest.py
and decorated appropriately (@pytest.fixture(scope="module")), without breaking this test
"""
def spill_ex():
    """
    example point release spill with 5 particles for testing
    """
    num_le = 5
    start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
    start_pos += (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time, persist=-1)
    return pSpill


class TestWindMover:
    """
    gnome.WindMover() test

    TODO: Move it to separate file
    """
    time_step = 15 * 60 # seconds
    spill = spill_ex()
    
    model_time = time_utils.sec_to_date(time_utils.date_to_sec(spill.release_time) + 1)

    time_val = np.zeros((1,), dtype=basic_types.datetime_value_2d)  # value is given as (r,theta)
    time_val['time']  = np.datetime64( spill.release_time.isoformat() )
    time_val['value'] = (2., 25.)
    wind = weather.Wind(timeseries=time_val)
    wm = movers.WindMover(wind)

    def test_string_repr_no_errors(self):
        print
        print "======================"
        print "repr(WindMover): "
        print repr( self.wm)
        print
        print "str(WindMover): "
        print str(self.wm)
        assert True

    def test_id_matches_builtin_id(self):
        assert id(self.wm) == self.wm.id
 
    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """
        self.spill.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.prepare_for_model_step(self.model_time, self.time_step)
 
        for ix in range(2):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            delta = self.wm.get_move(self.spill, self.time_step, curr_time)
            actual = self._expected_move()
            
            # the results should be independent of model time
            tol = 1e-8
            np.testing.assert_allclose(delta, actual, tol, tol,
                                       "WindMover.get_move() is not within a tolerance of " + str(tol), 0)
            
            print "Time step [sec]: \t" + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            print "C++ delta-move: " ; print str(delta)
            print "Expected delta-move: "; print str(actual)
                   
    def test_update_wind_vel(self):
        self.time_val['value'] = (1., 120.) # now given as (r, theta)
        self.wind.set_timeseries( self.time_val)
        self.test_get_move()
    
    def _expected_move(self):
        """
        Put the expected move logic in separate (fixture) if it gets used multiple times
        """
        # expected move
        uv = transforms.r_theta_to_uv_wind(self.time_val['value'])
        exp = np.zeros( (self.spill.num_LEs, 3) )
        exp[:,0] = self.spill['windages']*uv[0,0]*self.time_step # 'u'
        exp[:,1] = self.spill['windages']*uv[0,1]*self.time_step # 'v'
  
        xform = projections.FlatEarthProjection.meters_to_lonlat(exp, self.spill['positions'])
        return xform
