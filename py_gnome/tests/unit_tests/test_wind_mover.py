import os
from gnome import movers

from gnome import basic_types, weather
from gnome.spill_container import TestSpillContainer
from gnome.utilities import time_utils, transforms, convert
from gnome.utilities import projections

import numpy as np

from datetime import timedelta, datetime
import pytest

from hazpy import unit_conversion

datadir = os.path.join(os.path.dirname(__file__), r'SampleData')

def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with pytest.raises(TypeError):
        movers.WindMover()

    with pytest.raises(ValueError):
        file_ = r"SampleData/WindDataFromGnome.WND"
        wind = weather.Wind(file=file_)
        now = datetime.now()
        movers.WindMover(wind, active_start=now, active_stop=now)

# tolerance for np.allclose(..) function
atol = 1e-14
rtol = 1e-14

def test_read_file_init():
    """
    initialize from a long wind file
    """
    file_ = r"SampleData/WindDataFromGnome.WND"
    wind = weather.Wind(file=file_)
    wm = movers.WindMover(wind)
    wind_ts = wind.get_timeseries(data_format=basic_types.data_format.wind_uv, units='meter per second')
    _defaults(wm)   # check defaults set correctly
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)

    # make sure default user_units is correct and correctly called
    # NOTE: Following functionality is already tested in test_wind.py, but what the heck - do it here too.
    wind_ts = wind.get_timeseries(data_format=basic_types.data_format.wind_uv)
    cpp_timeseries['value'] = unit_conversion.convert('Velocity','meter per second',wind.user_units,cpp_timeseries['value'])
    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)
   
def test_timeseries_init(wind_circ):
    """
    test default properties of the object are initialized correctly
    """
    wm = movers.WindMover(wind_circ['wind'])
    _defaults(wm)
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'], atol, rtol)
   
   
def test_properties(wind_circ):
    """
    test setting the properties of the object
    """
    wm = movers.WindMover(wind_circ['wind'])
    
    wm.uncertain_duration = 1
    wm.uncertain_time_delay = 2
    wm.uncertain_speed_scale = 3
    wm.uncertain_angle_scale = 4
    
    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4    
   
   
def test_update_wind(wind_circ):
    """
    create a wind object and update it's timeseries. Make sure the internal C++ WindMover's properties have also changed
    """
    o_wind = wind_circ['wind']      # original wind value
    wm  = movers.WindMover(o_wind)  # define wind mover
    
    # update wind timeseries - default data_format is magnitude_direction
    t_dtv = np.zeros((3,), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    t_dtv.time = [datetime(2012,11,06,20,0+i,30) for i in range(3)]
    t_dtv.value= np.random.uniform(1,5, (3,2) )
    o_wind.set_timeseries(t_dtv, units='meters per second', data_format=basic_types.data_format.wind_uv)
    
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == t_dtv.time)
    assert np.allclose(cpp_timeseries['value'], t_dtv.value, atol, rtol)
    
    # set the wind timeseries back to test fixture values
    o_wind.set_timeseries(wind_circ['rq'], units='meters per second')
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'], atol, rtol)
  
def spill_ex():
    """
    example point release spill with 5 particles for testing
    """
    num_le = 5
    #start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    #pSpill = TestSpillContainer(num_le, start_pos, rel_time, persist=-1)
    #fixme: what to do about persistance?
    pSpill = TestSpillContainer(num_le, start_pos, rel_time)
    return pSpill

class TestWindMover:
    """
    gnome.WindMover() test
    """
    def __init__(self):
        #time_step = 15 * 60 # seconds
        self.spill = spill_ex()
        rel_time = self.spill.spills[0].release_time # digging a bit deep...
        #model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
        
        time_val = np.array((rel_time, (2., 25.)), dtype=basic_types.datetime_value_2d).reshape(1,)
        wind = weather.Wind(timeseries=time_val, units='meters per second')
        self.wm = movers.WindMover(wind)

    def test_string_repr_no_errors(self):
        print
        print "======================"
        print "repr(WindMover): "
        print repr( self.wm)
        print
        print "str(WindMover): "
        print str(self.wm)
        assert True

    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """
        self.spill.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.prepare_for_model_step( self.spill, self.time_step, self.model_time)

        for ix in range(2):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            delta = self.wm.get_move(self.spill, self.time_step, curr_time)
            actual = self._expected_move()
            
            # the results should be independent of model time
            tol = 1e-8
            np.testing.assert_allclose(delta, actual, tol, tol,
                                       "WindMover.get_move() is not within a tolerance of " + str(tol), 0)
            
            assert self.wm.active == True
            
            print "Time step [sec]: \t" + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            print "C++ delta-move: " ; print str(delta)
            print "Expected delta-move: "; print str(actual)

    def test_get_move_exceptions(self):
        curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step))
        tmp_windages = self.spill._data_arrays['windages']
        del self.spill._data_arrays['windages']
        with pytest.raises(KeyError):
            self.wm.get_move(self.spill, self.time_step, curr_time)
        self.spill._data_arrays['windages'] = tmp_windages

    def test_update_wind_vel(self):
        self.time_val['value'] = (1., 120.) # now given as (r, theta)
        self.wind.set_timeseries( self.time_val, units='meters per second')
        self.test_get_move()
        self.test_get_move_exceptions()
   
    def _expected_move(self):
        """
        Put the expected move logic in separate (fixture) if it gets used multiple times
        """
        # expected move
        uv = transforms.r_theta_to_uv_wind(self.time_val['value'])
        exp = np.zeros( (self.spill.num_elements, 3) )
        exp[:,0] = self.spill['windages']*uv[0,0]*self.time_step # 'u'
        exp[:,1] = self.spill['windages']*uv[0,1]*self.time_step # 'v'
        
        xform = projections.FlatEarthProjection.meters_to_lonlat(exp, self.spill['positions'])
        return xform

def test_timespan():
    """
    Ensure the active flag is being set correctly and checked, such that if active=False, the delta produced by get_move = 0
    """
    time_step = 15 * 60 # seconds

    #todo: hack for now, but should try to use same spill for all tests
    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    #fixme: what to do about persistance?
    spill = TestSpillContainer(5, start_pos, rel_time)
    spill.release_elements(datetime.now())

    model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
    spill.prepare_for_model_step(model_time, time_step)   # release particles

    time_val = np.zeros((1,), dtype=basic_types.datetime_value_2d)  # value is given as (r,theta)
    time_val['time']  = np.datetime64( rel_time.isoformat() )
    time_val['value'] = (2., 25.)
    wind = weather.Wind(timeseries=time_val, units='meters per second')

    wm = movers.WindMover(wind, active_start=model_time+timedelta(seconds=time_step))
    wm.prepare_for_model_step(spill, time_step, model_time)
    delta = wm.get_move(spill, time_step, model_time)
    assert wm.active == False
    assert np.all(delta == 0)   # model_time + time_step = active_start

    wm.active_start = model_time + timedelta(seconds=time_step/2)
    wm.prepare_for_model_step(spill, time_step, model_time)
    delta = wm.get_move(spill, time_step, model_time)
    assert wm.active == True
    assert np.all(delta[:,:2] != 0)   # model_time + time_step > active_start


#
#Helper methods for this module
#
def _defaults(wm):
    """
    checks the default properties of the WindMover object as given in the input are as expected
    """
    assert wm.active == True  # timespan is as big as possible
    assert wm.uncertain_duration == 10800
    assert wm.uncertain_time_delay == 0
    assert wm.uncertain_speed_scale == 2
    assert wm.uncertain_angle_scale == 0.4

def _get_timeseries_from_cpp(windmover):
    """
    local method for tests - returns the timeseries used internally by the C++ WindMover_c object.
    This should be the same as the timeseries stored in the self.wind object

    Data is returned as a datetime_value_2d array in units of meters per second in 
    data_format = wind_uv

    This is simply used for testing.
    """
    dtv = windmover.wind.get_timeseries(data_format=basic_types.data_format.wind_uv)
    tv  = convert.to_time_value_pair(dtv, basic_types.data_format.wind_uv)
    val = windmover.mover.get_time_value(tv['time'])
    tv['value']['u'] = val['u']
    tv['value']['v'] = val['v']

    return convert.to_datetime_value_2d( tv, basic_types.data_format.wind_uv)

def _assert_timeseries_equivalence(cpp_timeseries, wind_ts):
    """
    private method used to print data and assert 
    """
    print
    print "====================="
    print "WindMover timeseries [time], [u, v]: "
    print cpp_timeseries['time']
    print cpp_timeseries['value']
    print "---------------------"
    print "Wind timeseries [time], [u, v]: "
    print wind_ts['time']
    print wind_ts['value']

    assert np.all(cpp_timeseries['time'] == wind_ts['time'])
    assert np.allclose(cpp_timeseries['value'], wind_ts['value'], atol, rtol)
