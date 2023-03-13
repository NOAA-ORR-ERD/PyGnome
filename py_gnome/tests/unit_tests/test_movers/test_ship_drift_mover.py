'''
Test all operations for ship_drift mover work
'''





import datetime
import os

import numpy as np
import pytest

from gnome.movers import ShipDriftMover
from gnome.utilities import time_utils

from ..conftest import sample_sc_release, testdata


wind_file = testdata['c_GridWindMover']['wind_curv']
topology_file = testdata['c_GridWindMover']['top_curv']


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    with pytest.raises(TypeError):
        ShipDriftMover()

    with pytest.raises(ValueError):
        'file not found'
        ShipDriftMover(os.path.join('./', 'WindSpeedDirSubset.CUR'))

    with pytest.raises(TypeError):
        """
        note: you can pass in in int -- os.path.exists takes an integer file
              descriptor
        """
        ShipDriftMover(wind_file, topology_file=10.0)


def test_string_repr_no_errors():
    nw = ShipDriftMover(wind_file, topology_file, grid_type=2)
    print()
    print('======================')
    print('repr(ShipDriftMover): ')
    print(repr(nw))
    print()
    print('str(ShipDriftMover): ')
    print(str(nw))
    assert True


num_le = 4
start_pos = (-123.57152, 37.369436, 0.0)
rel_time = datetime.datetime(2006, 3, 31, 21, 0)
time_step = 30 * 60  # seconds
model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))


def test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    wind = ShipDriftMover(wind_file, topology_file, grid_type=2)
    delta = _certain_loop(pSpill, wind)

    _assert_move(delta)

    # set windage to be constant or each particle has a different position,
    # doesn't work with uncertainty on
    # assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    # assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs

    # returned delta is used in test_certain_uncertain test

def run_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    wind = ShipDriftMover(wind_file, topology_file, grid_type=2)
    delta = _certain_loop(pSpill, wind)

    # set windage to be constant or each particle has a different position,
    # doesn't work with uncertainty on
    # assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    # assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs

    # returned delta is used in test_certain_uncertain test
    return delta

def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = ShipDriftMover(wind_file, topology_file, grid_type=2)
    u_delta = _uncertain_loop(pSpill, wind)

    _assert_move(u_delta)

    # returned delta is used in test_certain_uncertain test
    # return u_delta

def run_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = ShipDriftMover(wind_file, topology_file, grid_type=2)
    u_delta = _uncertain_loop(pSpill, wind)

    return u_delta

def test_certain_uncertain():
    """
    make sure certain and uncertain loop results in different deltas
    """

    delta = run_loop()
    u_delta = run_uncertain_loop()
    print()
    print(delta)
    print(u_delta)
    assert np.all(delta[:, :2] != u_delta[:, :2])
    assert np.all(delta[:, 2] == u_delta[:, 2])


w_grid = ShipDriftMover(wind_file, topology_file, grid_type=2)


def test_default_props():
    """
    test default properties
    use _defaults helper function defined in test_wind_mover.py
    """
    assert w_grid.wind_scale == 1
    assert w_grid.extrapolate is False
    assert w_grid.time_offset == 0
    assert w_grid.active is True
    assert w_grid.drift_angle == 0


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print()
    print(delta)
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, wind):
    wind.prepare_for_model_run()
    wind.prepare_for_model_step(pSpill, time_step, model_time)
    delta = wind.get_move(pSpill, time_step, model_time)
    wind.model_step_is_done()

    return delta


def _uncertain_loop(pSpill, wind):
    wind.prepare_for_model_run()
    wind.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = wind.get_move(pSpill, time_step, model_time)
    wind.model_step_is_done()

    return u_delta


def test_serialize_deserialize():
    """
    test to_dict function for ShipDrift object
    create a new ship_drift object and make sure it has same properties
    """

    new_wind = ShipDriftMover(wind_file, topology_file, grid_type=2)
    serial = new_wind.serialize()
    nw2 = ShipDriftMover.deserialize(serial)

    assert new_wind == nw2

