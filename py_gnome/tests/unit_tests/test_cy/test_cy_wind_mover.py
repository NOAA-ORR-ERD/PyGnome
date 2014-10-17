#!/usr/bin/env python

"""
unit tests for cy_wind_mover wrapper

designed to be run with py.test
"""
import os

import numpy
np = numpy

from gnome.basic_types import (spill_type, ts_format,
                               velocity_rec,
                               time_value_pair,
                               world_point,
                               world_point_type)
from gnome.utilities import projections

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_wind_mover import CyWindMover

import cy_fixtures

datadir = os.path.join(os.path.dirname(__file__), r"sample_data")


def test_init():
    # can we create a wind_mover?
    wm = CyWindMover()

    assert wm.uncertain_duration == 10800
    assert wm.uncertain_time_delay == 0
    assert wm.uncertain_speed_scale == 2
    assert wm.uncertain_angle_scale == 0.4


def test_properties():
    wm = CyWindMover()

    wm.uncertain_duration = 1
    wm.uncertain_time_delay = 2
    wm.uncertain_speed_scale = 3
    wm.uncertain_angle_scale = 4

    assert wm.uncertain_duration == 1
    assert wm.uncertain_time_delay == 2
    assert wm.uncertain_speed_scale == 3
    assert wm.uncertain_angle_scale == 4


def test_eq():
    wm = CyWindMover()

    other_wm = CyWindMover()
    assert wm == other_wm

    other_wm = CyWindMover()
    other_wm.uncertain_duration = 1
    assert wm != other_wm

    other_wm = CyWindMover()
    other_wm.uncertain_time_delay = 2
    assert wm != other_wm

    other_wm = CyWindMover()
    other_wm.uncertain_speed_scale = 3
    assert wm != other_wm

    other_wm = CyWindMover()
    other_wm.uncertain_angle_scale = 4
    assert wm != other_wm


# use following constant wind for testing
const_wind = np.zeros((1, ), dtype=velocity_rec)
const_wind['u'] = 50  # meters per second?
const_wind['v'] = 100


class ConstantWind(cy_fixtures.CyTestMove):
    """
    Wind Mover object instantiated with a constant wind using member method
    set_constant_wind(...)
    Used for test setup
    """
    wm = CyWindMover()

    def __init__(self):
        super(ConstantWind, self).__init__()
        self.const_wind = const_wind
        self.wm.set_constant_wind(const_wind['u'], const_wind['v'])

    def test_move(self):
        """ forecast move """
        self.wm.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.get_move(self.model_time, self.time_step,
                         self.ref,
                         self.delta,
                         self.windage,
                         self.status,
                         spill_type.forecast)

    def test_move_uncertain(self):
        """ uncertain LEs """
        self.wm.prepare_for_model_step(self.model_time, self.time_step,
                                       len(self.spill_size), self.spill_size)
        self.wm.get_move(self.model_time, self.time_step,
                         self.ref,
                         self.u_delta,
                         self.windage,
                         self.status,
                         spill_type.uncertainty)


class ConstantWindWithOSSM(cy_fixtures.CyTestMove):
    """
    This defines the OSSMTimeValue_c object using the CyOSSMTime class,
    then uses the set_ossm method of CyWindMover object to set the
    time_dep member of the underlying WindMover_c C++ object

    Used for test setup
    """
    wm = CyWindMover()

    def __init__(self):
        super(ConstantWindWithOSSM, self).__init__()

        time_val = np.empty((1, ), dtype=time_value_pair)
        time_val['time'] = 0  # should not matter
        time_val['value'] = const_wind

        self.ossm = CyOSSMTime(timeseries=time_val)
        self.wm.set_ossm(self.ossm)

    def test_move(self):
        """
        invoke get_move (no uncertainty)
        """
        self.wm.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.get_move(self.model_time, self.time_step,
                         self.ref,
                         self.delta,
                         self.windage,
                         self.status,
                         spill_type.forecast)

    def test_move_uncertain(self):
        """ invoke get_move (uncertain LEs) """
        self.wm.prepare_for_model_step(self.model_time, self.time_step,
                                       len(self.spill_size), self.spill_size)
        self.wm.get_move(self.model_time,
                         self.time_step,
                         self.ref,
                         self.u_delta,
                         self.windage,
                         self.status,
                         spill_type.uncertainty)


class TestConstantWind:

    msg = '{0} is not within a tolerance of {1}'
    cw = ConstantWind()
    cw.test_move()

    cww_ossm = ConstantWindWithOSSM()
    cww_ossm.test_move()

    def test_constant_wind(self):
        """
        The result of get_move should be the same irrespective of
        whether we use OSSM time object or the fConstantValue member
        of the CyWindMover object
        Use the setup in ConstantWind and ConstantWindWithOSSM for this test
        """
        # the move should be the same from both objects

        print
        print 'test_constant_wind'
        print self.cw.delta
        print self.cww_ossm.delta
        np.testing.assert_equal(self.cw.delta, self.cww_ossm.delta,
                                'test_constant_wind() failed', 0)

    def test_move_value(self):
        # meters_per_deg_lat = 111120.00024
        # self.cw.windage/meters_per_deg_lat

        delta = np.zeros((self.cw.num_le, 3))
        delta[:, 0] = (self.cw.windage * self.cw.const_wind['u'] *
                       self.cw.time_step)
        delta[:, 1] = (self.cw.windage * self.cw.const_wind['v'] *
                       self.cw.time_step)

        ref = self.cw.ref.view(dtype=world_point_type).reshape((-1, 3))
        xform = projections.FlatEarthProjection.meters_to_lonlat(delta, ref)

        actual = np.zeros((self.cw.num_le, ), dtype=world_point)
        actual['lat'] = xform[:, 1]
        actual['long'] = xform[:, 0]
        tol = 1e-10

        np.testing.assert_allclose(self.cw.delta['lat'], actual['lat'],
                                   tol, tol,
                                   self.msg.format('get_time_value', tol),
                                   0)
        np.testing.assert_allclose(self.cw.delta['long'], actual['long'],
                                   tol, tol,
                                   self.msg.format('get_time_value', tol),
                                   0)

    def test_move_uncertain(self):
        self.cw.test_move_uncertain()
        self.cww_ossm.test_move_uncertain()
        print '=================================================='
        print ' Check move for uncertain LEs (test_move_uncertain)  '
        print '--- ConstandWind ------'
        print 'Forecast LEs delta: '
        print self.cw.delta
        print 'Uncertain LEs delta: '
        print self.cw.u_delta
        print '--- ConstandWind with OSSM ------'
        print 'Forecast LEs delta: '
        print self.cww_ossm.delta
        print 'Uncertain LEs delta: '
        print self.cww_ossm.u_delta
        assert np.all(self.cw.delta != self.cw.u_delta)
        assert np.all(self.cww_ossm.delta != self.cww_ossm.u_delta)


class TestVariableWind:
    """
    Uses OSSMTimeValue_c to define a variable wind
    - variable wind has 'v' component, so movement
      should only be in 'lat' direction of world point

    Leave as a class as we may add more methods to it for testing
    """
    wm = CyWindMover()
    cm = cy_fixtures.CyTestMove()
    delta = np.zeros((cm.num_le, ), dtype=world_point)

    time_val = np.zeros((2, ), dtype=time_value_pair)
    (time_val['time'])[:] = np.add([0, 3600], cm.model_time)  # after 1 hour
    (time_val['value']['v'])[:] = [100, 200]

    # CyOSSMTime needs the same scope as CyWindMover because CyWindMover
    # uses the C++ pointer defined in CyOSSMTime.time_dep. This must be defined
    # for the scope of CyWindMover

    ossm = CyOSSMTime(timeseries=time_val)
    wm.set_ossm(ossm)

    def test_move(self):
        for x in range(0, 3):
            vary_time = x * 1800
            self.wm.prepare_for_model_step(self.cm.model_time + vary_time,
                                           self.cm.time_step)
            self.wm.get_move(self.cm.model_time + vary_time,
                             self.cm.time_step,
                             self.cm.ref,
                             self.delta,
                             self.cm.windage,
                             self.cm.status,
                             spill_type.forecast)

            print self.delta
            assert np.all(self.delta['lat'] != 0)
            assert np.all(self.delta['long'] == 0)
            assert np.all(self.delta['z'] == 0)


def test_LE_not_in_water():
    """
    Tests get_move returns 0 for LE's that have a status
    different from in_water
    """
    wm = CyWindMover()
    cm = cy_fixtures.CyTestMove()
    delta = np.zeros((cm.num_le, ), dtype=world_point)
    cm.status[:] = 0

    wm.prepare_for_model_step(cm.model_time, cm.time_step)
    wm.get_move(cm.model_time, cm.time_step,
                cm.ref,
                delta,
                cm.windage,
                cm.status,
                spill_type.forecast)

    assert np.all(delta['lat'] == 0)
    assert np.all(delta['long'] == 0)
    assert np.all(delta['z'] == 0)


def test_z_greater_than_0():
    """
    If z > 0, then the particle is below the surface and the wind
    does not act on it.
    As such, the get_move should return 0 for delta
    """
    cw = ConstantWind()
    (cw.ref['z'])[:2] = 2  # particles 0,1 are not on the surface

    cw.test_move()

    assert np.all((cw.delta['lat'])[0:2] == 0)
    assert np.all((cw.delta['long'])[0:2] == 0)
    assert np.all((cw.delta['z'])[0:2] == 0)

    # for particles in water, there is a non zero delta
    assert np.all((cw.delta['lat'])[2:] != 0)
    assert np.all((cw.delta['long'])[2:] != 0)
    assert np.all((cw.delta['z'])[2:] == 0)


class TestObjectSerialization:
    '''
        Test all the serialization and deserialization methods that are
        available to the CyOSSMTime object.
    '''
    ossmT = CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                       file_contains=ts_format.magnitude_direction)
    wm = CyWindMover()
    wm.set_ossm(ossmT)

    def test_repr(self):
        '''
            Test that the repr method produces a string capable of reproducing
            the object.
        '''
        # This works, but in order to properly eval the repr string, we need
        # the top level gnome module, as well as the numpy 'array' type in the
        # global name space.
        # So before we try it, we first need to import like this:
        import gnome
        from numpy import array

        new_wm = eval(repr(self.wm))

        assert new_wm == self.wm
        assert repr(new_wm) == repr(self.wm)

    def test_pickle(self):
        '''
            Test that the object can be pickled and unpickled
        '''
        import pickle

        new_wm = pickle.loads(pickle.dumps(self.wm))
        assert new_wm == self.wm
        assert repr(new_wm) == repr(self.wm)

if __name__ == '__main__':
    cw = TestConstantWind()
    cw.test_constant_wind()
    cw.test_move_value()
    cw.test_move_uncertain()
