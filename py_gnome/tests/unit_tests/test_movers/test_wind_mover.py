
from datetime import timedelta, datetime

from pytest import raises

import numpy as np
import pytest
import nucos

from gnome.basic_types import datetime_value_2d, ts_format

from gnome.utilities.projections import FlatEarthProjection
from gnome.utilities.time_utils import date_to_sec, sec_to_date
from gnome.utilities.inf_datetime import InfDateTime
from gnome.utilities.transforms import r_theta_to_uv_wind
from gnome.utilities import convert

from gnome.environment import Wind

from gnome.spills import surface_point_line_spill
from gnome.spill_container import SpillContainer
from gnome.spills.substance import NonWeatheringSubstance

from gnome.movers import (PointWindMover,
                          constant_point_wind_mover,
                          point_wind_mover_from_file)
from gnome.exceptions import ReferencedObjectNotSet

from ..conftest import sample_sc_release, testdata


# PointWindMover tests

file_ = testdata['timeseries']['wind_ts']
file2_ = testdata['timeseries']['wind_cardinal']
filekph_ = testdata['timeseries']['wind_kph']


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with raises(ReferencedObjectNotSet) as excinfo:
        wm = PointWindMover()
        wm.prepare_for_model_run()

    print(excinfo.value)

    with raises(TypeError):
        """
        violates duck typing so may want to remove. Though current PointWindMover's
        backend cython object looks for C++ OSSM object which is embedded in
        Wind object which is why this check was enforced. Can be
        re-evaluated if there is a need.
        """
        PointWindMover(wind=10)


# tolerance for np.allclose(..) function

atol = 1e-14
rtol = 1e-14


def test_read_file_init():
    """
    initialize from a long wind file
    """

    wind = Wind(filename=file_)
    wm = PointWindMover(wind)
    wind_ts = wind.get_wind_data(coord_sys='uv', units='meter per second')
    _defaults(wm)  # check defaults set correctly
    assert not wm.make_default_refs
    cpp_timeseries = _get_timeseries_from_cpp(wm)
    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)

    # make sure default units is correct and correctly called
    # NOTE: Following functionality is already tested in test_wind.py,
    #       but what the heck - do it here too.

    wind_ts = wind.get_wind_data(coord_sys=ts_format.uv)
    cpp_timeseries['value'] = nucos.convert('Velocity',
                                         'meter per second', wind.units,
                                         cpp_timeseries['value'])

    _assert_timeseries_equivalence(cpp_timeseries, wind_ts)


def test_timeseries_init(wind_circ):
    """
    test default properties of the object are initialized correctly
    """
    wm = PointWindMover(wind_circ['wind'])
    _defaults(wm)
    assert not wm.make_default_refs
    cpp_timeseries = _get_timeseries_from_cpp(wm)

    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'],
                       atol, rtol)


def test_empty_init():
    '''
    wind=None
    '''
    wm = PointWindMover()
    assert wm.make_default_refs

    _defaults(wm)
    #assert wm.name == 'PointWindMover'
    print(wm.validate())


def test_properties(wind_circ):
    """
    test setting the properties of the object
    """
    wm = PointWindMover(wind_circ['wind'])

    wm.uncertain_duration = 1
    wm.uncertain_time_delay = 2
    wm.uncertain_speed_scale = 3
    wm.uncertain_angle_scale = 4

    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4
    assert wm.data_start == datetime(2012, 11, 6, 20, 10)
    assert wm.data_stop == datetime(2012, 11, 6, 20, 15)


def test_data_start_stop(wind_circ):
    """
    test data_start / stop properties
    """
    wm = PointWindMover(wind_circ['wind'])
    assert wm.data_start == datetime(2012, 11, 6, 20, 10)
    assert wm.data_stop == datetime(2012, 11, 6, 20, 15)


def test_update_wind(wind_circ):
    """
    Create a wind object and update it's timeseries.
    Make sure the internal C++ PointWindMover's properties have also changed
    """
    o_wind = wind_circ['wind']  # original wind value
    wm = PointWindMover(o_wind)  # define wind mover

    # update wind timeseries - default format is magnitude_direction

    t_dtv = np.zeros((3, ), dtype=datetime_value_2d).view(dtype=np.recarray)
    t_dtv.time = [datetime(2012, 11, 0o6, 20, 0 + i, 30)
                  for i in range(3)]
    t_dtv.value = np.random.uniform(1, 5, (3, 2))

    o_wind.set_wind_data(t_dtv, units='meter per second', coord_sys='uv')

    cpp_timeseries = _get_timeseries_from_cpp(wm)

    assert np.all(cpp_timeseries['time'] == t_dtv.time)
    assert np.allclose(cpp_timeseries['value'], t_dtv.value, atol, rtol)

    # set the wind timeseries back to test fixture values
    o_wind.set_wind_data(wind_circ['rq'], units='meter per second')
    cpp_timeseries = _get_timeseries_from_cpp(wm)

    assert np.all(cpp_timeseries['time'] == wind_circ['uv']['time'])
    assert np.allclose(cpp_timeseries['value'], wind_circ['uv']['value'],
                       atol, rtol)


class TestPrepareForModelStep(object):
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec

    sc = sample_sc_release(5, (3., 6., 0.), model_time)
    sc['windage_persist'][:2] = -1

    def test_windages_updated(self):
        '''
            explicitly test to make sure:
            - windages are being updated for persistence != 0 and
            - windages are not being changed for persistance == -1
        '''
        wind = Wind(timeseries=np.array((self.model_time, (2., 25.)),
                                        dtype=datetime_value_2d).reshape(1),
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)
            old_windages = np.copy(self.sc['windages'])
            wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            mask = self.sc['windage_persist'] == -1
            assert np.all(self.sc['windages'][mask] == old_windages[mask])

            mask = self.sc['windage_persist'] > 0
            assert np.all(self.sc['windages'][mask] != old_windages[mask])

    def test_constant_wind_before_model_time(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A constant wind should extrapolate if it is out of bounds,
            so prepare_for_model_step() should not fail.

            We are testing that the wind extrapolates properly, so the
            windages should be updated in the same way as the in-bounds test
        '''
        wind_time = datetime(2012, 8, 19, 13)  # one day before model time

        wind = Wind(timeseries=np.array((wind_time, (2., 25.)),
                                        dtype=datetime_value_2d).reshape(1),
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)
            print('curr_time = ', curr_time)

            old_windages = np.copy(self.sc['windages'])
            wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            mask = self.sc['windage_persist'] == -1
            assert np.all(self.sc['windages'][mask] == old_windages[mask])

            mask = self.sc['windage_persist'] > 0
            assert np.all(self.sc['windages'][mask] != old_windages[mask])

    def test_constant_wind_after_model_time(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A constant wind should extrapolate if it is out of bounds,
            so prepare_for_model_step() should not fail.

            We are testing that the wind extrapolates properly, so the
            windages should be updated in the same way as the in-bounds test
        '''
        wind_time = datetime(2012, 8, 21, 13)  # one day after model time

        wind = Wind(timeseries=np.array((wind_time, (2., 25.)),
                                        dtype=datetime_value_2d).reshape(1),
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)
            print('curr_time = ', curr_time)

            old_windages = np.copy(self.sc['windages'])
            wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            mask = self.sc['windage_persist'] == -1
            assert np.all(self.sc['windages'][mask] == old_windages[mask])

            mask = self.sc['windage_persist'] > 0
            assert np.all(self.sc['windages'][mask] != old_windages[mask])

    def test_variable_wind_before_model_time(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A variable wind should not extrapolate if it is out of bounds,
            so prepare_for_model_step() should fail with an exception
            in this case.
        '''
        wind_time = datetime(2012, 8, 19, 13)  # one day before model time

        time_series = (np.zeros((3, ), dtype=datetime_value_2d)
                       .view(dtype=np.recarray))
        time_series.time = [sec_to_date(date_to_sec(wind_time) +
                                        self.time_step * i)
                            for i in range(3)]
        time_series.value = np.array(((2., 25.), (2., 25.), (2., 25.)))

        wind = Wind(timeseries=time_series.reshape(3),
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)

            with raises(RuntimeError):
                wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

    def test_variable_wind_after_model_time(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A variable wind should not extrapolate if it is out of bounds,
            so prepare_for_model_step() should fail with an exception
            in this case.
        '''
        wind_time = datetime(2012, 8, 21, 13)  # one day after model time

        time_series = (np.zeros((3, ), dtype=datetime_value_2d)
                       .view(dtype=np.recarray))
        time_series.time = [sec_to_date(date_to_sec(wind_time) +
                                        self.time_step * i)
                            for i in range(3)]
        time_series.value = np.array(((2., 25.), (2., 25.), (2., 25.)))

        wind = Wind(timeseries=time_series.reshape(3),
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)

            with raises(RuntimeError):
                wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

    def test_variable_wind_before_model_time_with_extrapolation(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A variable wind can extrapolate if it is configured to do so,
            so prepare_for_model_step() should succeed in this case.

            We are testing that the wind extrapolates properly, so the
            windages should be updated in the same way as the in-bounds test
        '''
        wind_time = datetime(2012, 8, 19, 13)  # one day before model time

        time_series = (np.zeros((3, ), dtype=datetime_value_2d)
                       .view(dtype=np.recarray))
        time_series.time = [sec_to_date(date_to_sec(wind_time) +
                                        self.time_step * i)
                            for i in range(3)]
        time_series.value = np.array(((2., 25.), (2., 25.), (2., 25.)))

        wind = Wind(timeseries=time_series.reshape(3),
                    extrapolation_is_allowed=True,
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)

            old_windages = np.copy(self.sc['windages'])
            wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            mask = self.sc['windage_persist'] == -1
            assert np.all(self.sc['windages'][mask] == old_windages[mask])

            mask = self.sc['windage_persist'] > 0
            assert np.all(self.sc['windages'][mask] != old_windages[mask])

    def test_variable_wind_after_model_time_with_extrapolation(self):
        '''
            test to make sure the wind mover is behaving properly with
            out-of-bounds winds.
            A variable wind can extrapolate if it is configured to do so,
            so prepare_for_model_step() should succeed in this case.

            We are testing that the wind extrapolates properly, so the
            windages should be updated in the same way as the in-bounds test
        '''
        wind_time = datetime(2012, 8, 21, 13)  # one day after model time

        time_series = (np.zeros((3, ), dtype=datetime_value_2d)
                       .view(dtype=np.recarray))
        time_series.time = [sec_to_date(date_to_sec(wind_time) +
                                        self.time_step * i)
                            for i in range(3)]
        time_series.value = np.array(((2., 25.), (2., 25.), (2., 25.)))

        wind = Wind(timeseries=time_series.reshape(3),
                    extrapolation_is_allowed=True,
                    units='meter per second')

        wm = PointWindMover(wind)
        wm.prepare_for_model_run()

        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)

            old_windages = np.copy(self.sc['windages'])
            wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            mask = self.sc['windage_persist'] == -1
            assert np.all(self.sc['windages'][mask] == old_windages[mask])

            mask = self.sc['windage_persist'] > 0
            assert np.all(self.sc['windages'][mask] != old_windages[mask])


class TestWindMover(object):
    """
    gnome.PointWindMover() test
    """
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
    sc = sample_sc_release(5, (3., 6., 0.), model_time)

    time_val = np.array((model_time, (2., 25.)),
                        dtype=datetime_value_2d).reshape(1)
    wind = Wind(timeseries=time_val, units='meter per second')
    wm = PointWindMover(wind)

    wm.prepare_for_model_run()

    def test_string_repr_no_errors(self):
        print()
        print('======================')
        print('repr(PointWindMover): ')
        print(repr(self.wm))
        print()
        print('str(PointWindMover): ')
        print(str(self.wm))
        assert True

    def test_get_move(self):
        """
        Test the get_move(...) results in PointWindMover match the expected delta
        """
        for ix in range(2):
            curr_time = sec_to_date(date_to_sec(self.model_time) +
                                    self.time_step * ix)
            self.wm.prepare_for_model_step(self.sc, self.time_step, curr_time)

            delta = self.wm.get_move(self.sc, self.time_step, curr_time)
            actual = self._expected_move()

            # the results should be independent of model time
            tol = 1e-8

            msg = ('{0} is not within a tolerance of '
                   '{1}'.format('PointWindMover.get_move()', tol))
            np.testing.assert_allclose(delta, actual, tol, tol, msg, 0)

            assert self.wm.active

            ts = date_to_sec(curr_time) - date_to_sec(self.model_time)
            print(('Time step [sec]:\t{0}'
                   'C++ delta-move:\n{1}'
                   'Expected delta-move:\n{2}'
                   ''.format(ts, delta, actual)))

        self.wm.model_step_is_done()

    def test_get_move_exceptions(self):
        curr_time = sec_to_date(date_to_sec(self.model_time) + self.time_step)
        tmp_windages = self.sc._data_arrays['windages']
        del self.sc._data_arrays['windages']

        with raises(KeyError):
            self.wm.get_move(self.sc, self.time_step, curr_time)

        self.sc._data_arrays['windages'] = tmp_windages

    def test_update_wind_vel(self):
        self.time_val['value'] = (1., 120.)  # now given as (r, theta)
        self.wind.set_wind_data(self.time_val, units='meter per second')
        self.test_get_move()
        self.test_get_move_exceptions()

    def _expected_move(self):
        """
        Put the expected move logic in separate (fixture) if it gets used
        multiple times
        """
        uv = r_theta_to_uv_wind(self.time_val['value'])
        exp = np.zeros((self.sc.num_released, 3))
        exp[:, 0] = self.sc['windages'] * uv[0, 0] * self.time_step
        exp[:, 1] = self.sc['windages'] * uv[0, 1] * self.time_step

        xform = FlatEarthProjection.meters_to_lonlat(exp, self.sc['positions'])
        return xform

#incompatible with one substance per spill container
@pytest.mark.xfail()
def test_windage_index():
    """
    A very simple test to make sure windage is set for the correct sc
    if staggered release
    """
    sc = SpillContainer()
    rel_time = datetime(2013, 1, 1, 0, 0)
    timestep = 30
    for i in range(2):
        spill = surface_point_line_spill(num_elements=5,
                                         start_position=(0., 0., 0.),
                                         release_time=rel_time + i * timedelta(hours=1),
                                         substance=NonWeatheringSubstance(windage_range=(i * .01 +
                                                               .01, i * .01 + .01),
                                                               windage_persist=900)
                                         )
        sc.spills.add(spill)

    #windage = ['windages', 'windage_range', 'windage_persist']
    sc.prepare_for_model_run(array_types=spill.array_types)
    sc.release_elements(timestep, rel_time)

    wm = constant_point_wind_mover(5, 0)
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
                assert np.all(sc['windages'][mask] ==
                              sp.substance.windage_range[0])

    # only 1st spill is released
    _check_index(sc)  # 1st ASSERT

    sc.release_elements(timestep, rel_time + timedelta(hours=1))
    wm.prepare_for_model_step(sc, timestep, rel_time)
    _check_index(sc)  # 2nd ASSERT


def test_timespan():
    """
    Ensure the active flag is being set correctly and checked,
    such that if active=False, the delta produced by get_move = 0
    """
    time_step = 15 * 60  # seconds

    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec

    sc = sample_sc_release(5, start_pos, rel_time)

    # value is given as (r,theta)
    model_time = rel_time
    time_val = np.zeros((1, ), dtype=datetime_value_2d)
    time_val['time'] = rel_time
    time_val['value'] = (2., 25.)

    wm = PointWindMover(Wind(timeseries=time_val,
                        units='meter per second'),
                   active_range=(model_time + timedelta(seconds=time_step),
                                 InfDateTime('inf')))

    wm.prepare_for_model_run()
    wm.prepare_for_model_step(sc, time_step, model_time)

    delta = wm.get_move(sc, time_step, model_time)
    wm.model_step_is_done()

    assert wm.active is False
    assert np.all(delta == 0)  # model_time + time_step = active_start

    wm.active_range = (model_time - timedelta(seconds=time_step / 2),
                       InfDateTime('inf'))
    wm.prepare_for_model_step(sc, time_step, model_time)

    delta = wm.get_move(sc, time_step, model_time)
    wm.model_step_is_done()

    assert wm.active is True
    print('''\ntest_timespan delta \n{0}'''.format(delta))
    assert np.all(delta[:, :2] != 0)  # model_time + time_step > active_start


def test_active():
    """ test that mover must be both active and on to get movement """

    time_step = 15 * 60  # seconds

    start_pos = (3., 6., 0.)
    rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec

    sc = sample_sc_release(5, start_pos, rel_time)

    # value is given as (r,theta)

    time_val = np.zeros((1, ), dtype=datetime_value_2d)
    time_val['time'] = rel_time
    time_val['value'] = (2., 25.)

    wm = PointWindMover(Wind(timeseries=time_val, units='meter per second'),
                   on=False)

    wm.prepare_for_model_run()
    wm.prepare_for_model_step(sc, time_step, rel_time)

    delta = wm.get_move(sc, time_step, rel_time)
    wm.model_step_is_done()

    assert wm.active is False
    assert np.all(delta == 0)  # model_time + time_step = active_start


def test_constant_point_wind_mover():
    """
    tests the constant_point_wind_mover utility function
    """
    with raises(nucos.InvalidUnitError):
        _wm = constant_point_wind_mover(10, 45, units='some_random_string')

    wm = constant_point_wind_mover(10, 45, units='m/s')

    sc = sample_sc_release(1)

    time_step = 1000
    model_time = datetime(2013, 3, 1, 0)

    wm.prepare_for_model_step(sc, time_step, model_time)
    delta = wm.get_move(sc, time_step, model_time)

    # 45 degree wind at the equator -- u,v should be the same
    assert delta[0][0] == delta[0][1]


def test_constant_point_wind_mover_bounds():
    wm = constant_point_wind_mover(10, 45, units='knots')

    assert wm.data_start == wm.data_stop


def test_point_wind_mover_from_file():
    wm = point_wind_mover_from_file(file_)
    print(wm.wind.filename)
    assert wm.wind.filename == file_


def test_point_wind_mover_from_file_cardinal():
    wm = point_wind_mover_from_file(file2_)
    print(wm.wind.filename)
    assert wm.wind.filename == file2_


def test_point_wind_mover_from_file_kph_units():
    wm = point_wind_mover_from_file(filekph_)
    print(wm.wind.filename)
    assert wm.wind.filename == filekph_


def test_serialize_deserialize(wind_circ):
    """
    tests and illustrate the funcitonality of serialize/deserialize for
    PointWindMover.
    """
    wind = Wind(filename=file_)
    wm = PointWindMover(wind)
    serial = wm.serialize()
    assert 'wind' in serial

    wm2 = wm.deserialize(serial)

    assert wm == wm2


# @pytest.mark.parametrize("save_ref", [False, True])
# def test_save_load(save_ref, saveloc_):
#     """
#     tests and illustrates the functionality of save/load for
#     PointWindMover
#     """
#     wind = Wind(filename=file_)
#     wm = PointWindMover(wind)
#     wm_fname = 'WindMover_save_test.json'
#     refs = None
#     if save_ref:
#         w_fname = 'Wind.json'
#         refs = References()
#         refs.reference(wind, w_fname)
#         wind.save(saveloc_, refs, w_fname)
#
#     wm.save(saveloc_, references=refs, filename=wm_fname)
#
#     l_refs = References()
#     obj = load(os.path.join(saveloc_, wm_fname), l_refs)
#     assert (obj == wm and obj is not wm)
#     assert (obj.wind == wind and obj.wind is not wind)
#     shutil.rmtree(saveloc_)  # clean-up


def test_array_types():
    """
    Check the array_types property of PointWindMover contains array_types.PointWindMover
    """
    # PointWindMover does not modify Wind object!
    wm = PointWindMover(Wind(filename=file_))

    for t in ('windages', 'windage_range', 'windage_persist'):
        assert t in wm.array_types


def _defaults(wm):
    """
    checks the default properties of the PointWindMover object as given in the input
    are as expected
    """
    # timespan is as big as possible
    assert wm.active is True
    assert wm.uncertain_duration == 3.0
    assert wm.uncertain_time_delay == 0
    assert wm.uncertain_speed_scale == 2
    assert wm.uncertain_angle_scale == 0.4


def _get_timeseries_from_cpp(PointWindMover):
    """
    local method for tests - returns the timeseries used internally
    by the C++ WindMover_c object.
    This should be the same as the timeseries stored in the self.wind object

    Data is returned as a datetime_value_2d array in units of meter per second
    in format = uv

    This is simply used for testing.
    """
    dtv = PointWindMover.wind.get_wind_data(coord_sys=ts_format.uv)
    tv = convert.to_time_value_pair(dtv, ts_format.uv)
    val = PointWindMover.mover.get_time_value(tv['time'])

    tv['value']['u'] = val['u']
    tv['value']['v'] = val['v']

    return convert.to_datetime_value_2d(tv, ts_format.uv)


def _assert_timeseries_equivalence(cpp_timeseries, wind_ts):
    """
    private method used to print data and assert
    """
    print()
    print('=====================')
    print('PointWindMover timeseries [time], [u, v]: ')
    print(cpp_timeseries['time'])
    print(cpp_timeseries['value'])
    print('---------------------')
    print('Wind timeseries [time], [u, v]: ')
    print(wind_ts['time'])
    print(wind_ts['value'])

    assert np.all(cpp_timeseries['time'] == wind_ts['time'])
    assert np.allclose(cpp_timeseries['value'], wind_ts['value'], atol, rtol)
