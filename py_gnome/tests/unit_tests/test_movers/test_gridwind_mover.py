'''
Test all operations for gridcurrent mover work
'''




import os
import datetime

import numpy as np

import pytest

from gnome.utilities import time_utils
from gnome.movers import c_GridWindMover

from ..conftest import sample_sc_release, testdata
# default settings are the same for both objects
from .test_wind_mover import _defaults


wind_file = testdata['c_GridWindMover']['wind_curv']
topology_file = testdata['c_GridWindMover']['top_curv']


num_le = 4
start_pos = (-123.57152, 37.369436, 0.0)
rel_time = datetime.datetime(2006, 3, 31, 21, 0)
time_step = 30 * 60  # seconds
# fixme -- huh???
# model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))

model_time = rel_time


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    with pytest.raises(TypeError):
        # we need to supply at least a filename
        c_GridWindMover()

    with pytest.raises(ValueError):
        # wind file not found
        c_GridWindMover('bogus')

    with pytest.raises(ValueError):
        # topology file not found
        c_GridWindMover(wind_file, topology_file='bogus')

    with pytest.raises(TypeError):
        # topology file needs to be a string filename
        c_GridWindMover(wind_file, topology_file=10)


def test_init_defaults():
    gw = c_GridWindMover(wind_file)

    assert gw.name == os.path.split(wind_file)[1]
    assert gw.filename == wind_file
    assert gw.topology_file is None


def test_string_repr_no_errors():
    gw = c_GridWindMover(wind_file, topology_file)
    print()
    print('======================')
    print('repr(PointWindMover): ')
    print(repr(gw))
    print()
    print('str(PointWindMover): ')
    print(str(gw))

    # TODO, FIXME: We need a way of validating this if we really care what
    #              the str() and repr() methods are doing.
    assert True


def test_loop():
    """
    test one time step with no uncertainty on the spill
    - checks there is non-zero motion.
    - also checks the motion is same for all LEs

    - Uncertainty needs to be off.
    - Windage needs to be set to not vary or each particle will have a
      different position,  This is done by setting the windage range to have
      all the same values (min == max).
    """
    pSpill = sample_sc_release(num_elements=num_le,
                               start_pos=start_pos,
                               release_time=rel_time,
                               windage_range=(0.01, 0.01))

    wind = c_GridWindMover(wind_file, topology_file)

    delta = _certain_loop(pSpill, wind)
    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs

def run_loop():
    """
    test one time step with no uncertainty on the spill
    - checks there is non-zero motion.
    - also checks the motion is same for all LEs

    - Uncertainty needs to be off.
    - Windage needs to be set to not vary or each particle will have a
      different position,  This is done by setting the windage range to have
      all the same values (min == max).
    """
    pSpill = sample_sc_release(num_elements=num_le,
                               start_pos=start_pos,
                               release_time=rel_time,
                               windage_range=(0.01, 0.01))

    wind = c_GridWindMover(wind_file, topology_file)

    delta = _certain_loop(pSpill, wind)

    # returned delta is used in test_certain_uncertain test
    return delta

def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = c_GridWindMover(wind_file, topology_file)
    u_delta = _certain_loop(pSpill, wind)

    _assert_move(u_delta)

def run_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = c_GridWindMover(wind_file, topology_file)
    u_delta = _certain_loop(pSpill, wind)

    # returned delta is used in test_certain_uncertain test
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


w_grid = c_GridWindMover(wind_file, topology_file)


def test_default_props():
    """
    test default properties
    use _defaults helper function defined in test_wind_mover.py
    """
    # ==========================================================================
    # assert w_grid.active == True  # timespan is as big as possible
    # assert w_grid.uncertain_duration == 3.0
    # assert w_grid.uncertain_time_delay == 0
    # assert w_grid.uncertain_speed_scale == 2
    # assert w_grid.uncertain_angle_scale == 0.4
    # assert w_grid.uncertain_angle_units == 'rad'
    # ==========================================================================
    assert w_grid.wind_scale == 1
    assert w_grid.extrapolate is False
    assert w_grid.time_offset == 0

    _defaults(w_grid)


def test_uncertain_time_delay():
    """
    test setting / getting properties
    """

    w_grid.uncertain_time_delay = 3
    assert w_grid.uncertain_time_delay == 3


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

# _uncertain_loop = _certain_loop

# # fixme: why isn't this just the above -- it looks the same?
# def _uncertain_loop(pSpill, wind):
#     wind.prepare_for_model_run()
#     wind.prepare_for_model_step(pSpill, time_step, model_time)
#     u_delta = wind.get_move(pSpill, time_step, model_time)
#     wind.model_step_is_done()

#     return u_delta


def test_serialize_deserialize():
    """
    test to_dict function for GridWind object
    create a new grid_wind object and make sure it has same properties
    """

    grid_wind = c_GridWindMover(wind_file, topology_file)
    serial = grid_wind.serialize()
    gw2 = c_GridWindMover.deserialize(serial)

    assert grid_wind == gw2
