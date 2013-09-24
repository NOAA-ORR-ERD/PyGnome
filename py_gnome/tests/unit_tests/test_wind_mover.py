import os
from datetime import timedelta, datetime

import numpy as np
import pytest

from hazpy import unit_conversion

from gnome.basic_types import datetime_value_2d, ts_format

from gnome import environment
from gnome import array_types

from gnome.utilities.projections import FlatEarthProjection
from gnome.utilities.time_utils import date_to_sec, sec_to_date
from gnome.utilities.transforms import r_theta_to_uv_wind
from gnome.utilities import convert

from gnome.movers import WindMover, constant_wind_mover, \
    wind_mover_from_file

from gnome.spill import PointSourceSurfaceRelease
from gnome.spill_container import SpillContainer, TestSpillContainer

datadir = os.path.join(os.path.dirname(__file__), r'sample_data')
file_ = os.path.join(datadir, r'WindDataFromGnome.WND')


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """

    with pytest.raises(TypeError):
        WindMover()

    with pytest.raises(ValueError):
        wind = environment.Wind(filename=file_)
        now = datetime.now()
        WindMover(wind, active_start=now, active_stop=now)

    with pytest.raises(TypeError):
        WindMover(wind=10)


# tolerance for np.allclose(..) function

atol = 1e-14
rtol = 1e-14


def test_read_file_init():
    """
    initialize from a long wind file
    """

    wind = environment.Wind(filename=file_)
    wm = WindMover(wind)
    wind_ts = wind.get_timeseries(format='uv', units='meter per second')
    _defaults(wm)  # check defaults set correctly
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)

    # make sure default units is correct and correctly called
    # NOTE: Following functionality is already tested in test_wind.py,
    #       but what the heck - do it here too.

    wind_ts = wind.get_timeseries(format=ts_format.uv)
    cpp_timeseries['value'] = unit_conversion.convert('Velocity',
            'meter per second', wind.units, cpp_timeseries['value'])

    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)


def test_timeseries_init(wind_circ):
    """
    test default properties of the object are initialized correctly
    """

    wm = WindMover(wind_circ['wind'])
    _defaults(wm)
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'
                       ], atol, rtol)


def test_properties(wind_circ):
    """
    test setting the properties of the object
    """

    wm = WindMover(wind_circ['wind'])

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
    Create a wind object and update it's timeseries.
    Make sure the internal C++ WindMover's properties have also changed
    """

    o_wind = wind_circ['wind']  # original wind value
    wm = WindMover(o_wind)  # define wind mover

    # update wind timeseries - default format is magnitude_direction

    t_dtv = np.zeros((3, ),
                     dtype=datetime_value_2d).view(dtype=np.recarray)
    t_dtv.time = [datetime(
        2012,
        11,
        06,
        20,
        0 + i,
        30,
        ) for i in range(3)]
    t_dtv.value = np.random.uniform(1, 5, (3, 2))
    o_wind.set_timeseries(t_dtv, units='meter per second', format='uv')

    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == t_dtv.time)
    assert np.allclose(cpp_timeseries['value'], t_dtv.value, atol, rtol)

    # set the wind timeseries back to test fixture values

    o_wind.set_timeseries(wind_circ['rq'], units='meter per second')
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'
                       ], atol, rtol)


class TestWindMover:

    """
    gnome.WindMover() test
    """

    time_step = 15 * 60  # seconds
    rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
    sc = TestSpillContainer(5, (3., 6., 0.), rel_time)
    model_time = rel_time

    time_val = np.array((rel_time, (2., 25.)),
                        dtype=datetime_value_2d).reshape(1)
    wind = environment.Wind(timeseries=time_val,
                            units='meter per second')
    wm = WindMover(wind)
    wm.prepare_for_model_run()

    def test_string_repr_no_errors(self):
        print
        print '======================'
        print 'repr(WindMover): '
        print repr(self.wm)
        print
        print 'str(WindMover): '
        print str(self.wm)
        assert True

    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """

        self.sc.prepare_for_model_step(self.model_time)
        self.wm.prepare_for_model_step(self.sc, self.time_step,
                self.model_time)

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time)
                                    + self.time_step * ix)
            delta = self.wm.get_move(self.sc, self.time_step, curr_time)
            actual = self._expected_move()

            # the results should be independent of model time

            tol = 1e-8

            msg = '{0} is not within a tolerance of {1}'
            np.testing.assert_allclose(
                delta,
                actual,
                tol,
                tol,
                msg.format('WindMover.get_move()', tol),
                0,
                )

            assert self.wm.active == True

            ts = date_to_sec(curr_time) - date_to_sec(self.model_time)
            print 'Time step [sec]:\t%s' % ts
            print 'C++ delta-move:\n%s' % str(delta)
            print 'Expected delta-move:\n%s' % str(actual)

        self.wm.model_step_is_done()

    def test_get_move_exceptions(self):
        curr_time = sec_to_date(date_to_sec(self.model_time)
                                + self.time_step)
        tmp_windages = self.sc._data_arrays['windages']
        del self.sc._data_arrays['windages']
        with pytest.raises(KeyError):
            self.wm.get_move(self.sc, self.time_step, curr_time)
        self.sc._data_arrays['windages'] = tmp_windages

    def test_update_wind_vel(self):
        self.time_val['value'] = (1., 120.)  # now given as (r, theta)
        self.wind.set_timeseries(self.time_val, units='meter per second'
                                 )
        self.test_get_move()
        self.test_get_move_exceptions()

    def _expected_move(self):
        """
        Put the expected move logic in separate (fixture) if it gets used
        multiple times
        """

        # expected move

        uv = r_theta_to_uv_wind(self.time_val['value'])
        exp = np.zeros((self.sc.num_elements, 3))
        exp[:, 0] = self.sc['windages'] * uv[0, 0] * self.time_step  # 'u'
        exp[:, 1] = self.sc['windages'] * uv[0, 1] * self.time_step  # 'v'

        xform = FlatEarthProjection.meters_to_lonlat(exp,
                self.sc['positions'])
        return xform


def test_windage_index():
    """
    A very simple test to make sure windage is set for the correct sc
    if staggered release
    """

    sc = SpillContainer()
    rel_time = datetime(2013, 1, 1, 0, 0)
    timestep = 30
    for i in range(2):
        spill = PointSourceSurfaceRelease(num_elements=5,
                start_position=(0., 0., 0.), release_time=rel_time + i
                * timedelta(hours=1), windage_range=(i * .01 + .01, i
                * .01 + .01), windage_persist=900)
        sc.spills.add(spill)

    sc.prepare_for_model_run(rel_time, dict(array_types.windage))
    sc.release_elements(rel_time, timestep)
    wm = WindMover(environment.ConstantWind(5, 0))
    wm.prepare_for_model_step(sc, timestep, rel_time)
    wm.model_step_is_done()  # need this to toggle _windage_is_set_flag

    def _check_index(sc):
        '''
        internal function for doing the test after windage is set
        - called twice so made a function
        '''

        # only 1st sc is released

        for sp in sc.spills:
            mask = sc.get_spill_mask(sp)
            if np.any(mask):
                assert np.all(sc['windages'][mask]
                              == sp.windage_range[0])

    # only 1st sc is released

    _check_index(sc)  # 1st ASSERT

    sc.release_elements(rel_time + timedelta(hours=1), timestep)
    wm.prepare_for_model_step(sc, timestep, rel_time)
    _check_index(sc)  # 2nd ASSERT


def test_timespan():
    """
    Ensure the active flag is being set correctly and checked,
    such that if active=False, the delta produced by get_move = 0
    """

    time_step = 15 * 60  # seconds

    # todo: hack for now, but should try to use same sc for all tests

    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec

    # fixme: what to do about persistance?

    sc = TestSpillContainer(5, start_pos, rel_time)
    sc.release_elements(datetime.now(), time_step=100)

    # model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)

    model_time = rel_time
    sc.prepare_for_model_step(model_time)  # release particles

    # value is given as (r,theta)

    time_val = np.zeros((1, ), dtype=datetime_value_2d)
    time_val['time'] = np.datetime64(rel_time.isoformat())
    time_val['value'] = (2., 25.)

    wm = WindMover(environment.Wind(timeseries=time_val,
                   units='meter per second'), active_start=model_time
                   + timedelta(seconds=time_step))

    wm.prepare_for_model_run()
    wm.prepare_for_model_step(sc, time_step, model_time)
    delta = wm.get_move(sc, time_step, model_time)
    wm.model_step_is_done()
    assert wm.active == False
    assert np.all(delta == 0)  # model_time + time_step = active_start

    wm.active_start = model_time - timedelta(seconds=time_step / 2)
    wm.prepare_for_model_step(sc, time_step, model_time)
    delta = wm.get_move(sc, time_step, model_time)
    wm.model_step_is_done()

    assert wm.active == True
    print '''
 test_timespan: delta 
{0}'''.format(delta)
    assert np.all(delta[:, :2] != 0)  # model_time + time_step > active_start


def test_active():
    """ test that mover must be both active and on to get movement """

    time_step = 15 * 60  # seconds

    # todo: hack for now, but should try to use same sc for all tests

    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec

    # fixme: what to do about persistance?

    sc = TestSpillContainer(5, start_pos, rel_time)
    sc.release_elements(datetime.now(), time_step=100)

    # model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)

    model_time = rel_time
    sc.prepare_for_model_step(model_time)  # release particles

    # value is given as (r,theta)

    time_val = np.zeros((1, ), dtype=datetime_value_2d)
    time_val['time'] = np.datetime64(rel_time.isoformat())
    time_val['value'] = (2., 25.)

    wm = WindMover(environment.Wind(timeseries=time_val,
                   units='meter per second'), on=False)

    wm.prepare_for_model_run()
    wm.prepare_for_model_step(sc, time_step, model_time)
    delta = wm.get_move(sc, time_step, model_time)
    wm.model_step_is_done()
    assert wm.active == False
    assert np.all(delta == 0)  # model_time + time_step = active_start


def test_constant_wind_mover():
    """
    tests the constant_wind_mover utility function
    """

    with pytest.raises(Exception):

        # it should raise an InvalidUnitError, but I don't want to have to
        # import unit_conversion just for that...

        wm = constant_wind_mover(10, 45, units='some_random_string')

    wm = constant_wind_mover(10, 45, units='m/s')

    sc = TestSpillContainer(1)

    print wm
    print repr(wm.wind)
    print wm.wind.get_timeseries()
    time_step = 1000
    model_time = datetime(2013, 3, 1, 0)
    wm.prepare_for_model_step(sc, time_step, model_time)
    delta = wm.get_move(sc, time_step, model_time)
    print 'delta:', delta

    # 45 degree wind at the equator -- u,v should be the same

    assert delta[0][0] == delta[0][1]


def test_wind_mover_from_file():
    wm = wind_mover_from_file(file_)
    print wm.wind.filename
    assert wm.wind.filename == file_


def test_new_from_dict():
    """
    Currently only checks that new object can be created from dict
    It does not check equality of objects
    """

    wind = environment.Wind(filename=file_)
    wm = WindMover(wind)  # WindMover does not modify Wind object!
    wm_state = wm.to_dict('create')

    # must create a Wind object and add this to wm_state dict

    wind2 = environment.Wind.new_from_dict(wind.to_dict('create'))
    wm_state.update({'wind': wind2})
    wm2 = WindMover.new_from_dict(wm_state)

    # check serializable state is correct

    assert all([wm.__getattribute__(k) == wm2.__getattribute__(k)
               for k in WindMover.state.get_names('create') if k
               != 'wind_id' and k != 'obj_type'])
    assert wm.wind.id == wm2.wind.id


def test_exception_new_from_dict():

    # WindMover does not modify Wind object!

    wm = WindMover(environment.Wind(filename=file_))

    wm_state = wm.to_dict('create')
    wm_state.update({'wind': environment.Wind(filename=file_)})

    with pytest.raises(ValueError):
        WindMover.new_from_dict(wm_state)


def test_array_types():
    """
    Check the array_types property of WindMover contains array_types.windage
    """

    # WindMover does not modify Wind object!

    wm = WindMover(environment.Wind(filename=file_))
    wm_array = wm.array_types

    assert len(wm_array) == len(array_types.windage)

    for (key, val) in dict(array_types.windage).iteritems():
        assert key in wm_array
        assert wm_array[key] == val
        wm_array.pop(key)

    assert len(wm_array) == 0


def _defaults(wm):
    """
    checks the default properties of the WindMover object as given in the input
    are as expected
    """

    assert wm.active == True  # timespan is as big as possible
    assert wm.uncertain_duration == 10800
    assert wm.uncertain_time_delay == 0
    assert wm.uncertain_speed_scale == 2
    assert wm.uncertain_angle_scale == 0.4


def _get_timeseries_from_cpp(windmover):
    """
    local method for tests - returns the timeseries used internally
    by the C++ WindMover_c object.
    This should be the same as the timeseries stored in the self.wind object

    Data is returned as a datetime_value_2d array in units of meter per second
    in format = uv

    This is simply used for testing.
    """

    dtv = windmover.wind.get_timeseries(format=ts_format.uv)
    tv = convert.to_time_value_pair(dtv, ts_format.uv)
    val = windmover.mover.get_time_value(tv['time'])
    tv['value']['u'] = val['u']
    tv['value']['v'] = val['v']

    return convert.to_datetime_value_2d(tv, ts_format.uv)


def _assert_timeseries_equivalence(cpp_timeseries, wind_ts):
    """
    private method used to print data and assert
    """

    print
    print '====================='
    print 'WindMover timeseries [time], [u, v]: '
    print cpp_timeseries['time']
    print cpp_timeseries['value']
    print '---------------------'
    print 'Wind timeseries [time], [u, v]: '
    print wind_ts['time']
    print wind_ts['value']

    assert np.all(cpp_timeseries['time'] == wind_ts['time'])
    assert np.allclose(cpp_timeseries['value'], wind_ts['value'], atol,
                       rtol)


