'''
Test all operations for gridcurrent mover work
'''

import datetime
import os

import numpy as np
import pytest

from gnome.movers import GridCurrentMover
from gnome.utilities import time_utils
from gnome.utilities.remote_data import get_datafile
from gnome.persist import load

from conftest import sample_sc_release

here = os.path.dirname(__file__)
cur_dir = os.path.join(here, 'sample_data', 'currents')

curr_file = get_datafile(os.path.join(cur_dir, 'ChesBay.nc'))
topology_file = get_datafile(os.path.join(cur_dir, 'ChesBay.dat'))


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    bad_file = os.path.join(cur_dir, 'ChesBay.CUR')
    with pytest.raises(ValueError):
        GridCurrentMover(bad_file)

    bad_file = get_datafile(os.path.join(cur_dir, 'BigCombinedwMapBad.cur'))
    with pytest.raises(OSError):
        GridCurrentMover(bad_file)

    with pytest.raises(TypeError):
        GridCurrentMover(curr_file, topology_file=10)


num_le = 4
start_pos = (-76.149368, 37.74496, 0)
rel_time = datetime.datetime(2004, 12, 31, 13)
time_step = 15 * 60  # seconds
model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))


def test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    curr = GridCurrentMover(curr_file, topology_file)
    delta = _certain_loop(pSpill, curr)

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros

    return delta


def test_uncertain_loop(uncertain_time_delay=0):
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    curr = GridCurrentMover(curr_file, topology_file)
    curr.uncertain_time_delay=uncertain_time_delay
    u_delta = _uncertain_loop(pSpill, curr)

    _assert_move(u_delta)

    return u_delta


def test_certain_uncertain():
    """
    make sure certain and uncertain loop results in different deltas
    """

    delta = test_loop()
    u_delta = test_uncertain_loop()
    print
    print delta
    print u_delta
    assert np.all(delta[:, :2] != u_delta[:, :2])
    assert np.all(delta[:, 2] == u_delta[:, 2])
    uncertain_time_delay = 3
    u_delta = test_uncertain_loop(uncertain_time_delay)
    print u_delta
    assert np.all(delta[:, :2] == u_delta[:, :2])


c_grid = GridCurrentMover(curr_file,topology_file)


def test_default_props():
    """
    test default properties
    """

    assert c_grid.current_scale == 1
    assert c_grid.uncertain_time_delay == 0
    assert c_grid.uncertain_duration == 24
    assert c_grid.uncertain_cross == .25
    assert c_grid.uncertain_along == .5
    assert c_grid.extrapolate == False
    assert c_grid.time_offset == 0


def test_scale():
    """
    test setting / getting properties
    """

    c_grid.uncertain_time_delay = 3
    assert c_grid.uncertain_time_delay == 3


def test_scale_value():
    """
    test setting / getting properties
    """

    c_grid.current_scale = 2
    print c_grid.current_scale
    assert c_grid.current_scale == 2


def test_extrapolate():
    """
    test setting / getting properties
    """

    c_grid.extrapolate = True
    print c_grid.extrapolate
    assert c_grid.extrapolate == True


def test_offset_time():
    """
    test setting / getting properties
    """

    c_grid.time_offset = -8
    print c_grid.time_offset
    assert c_grid.time_offset == -8


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print
    print delta
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, curr):
    curr.prepare_for_model_run()
    curr.prepare_for_model_step(pSpill, time_step, model_time)
    delta = curr.get_move(pSpill, time_step, model_time)
    curr.model_step_is_done()

    return delta


def _uncertain_loop(pSpill, curr):
    curr.prepare_for_model_run()
    curr.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = curr.get_move(pSpill, time_step, model_time)
    curr.model_step_is_done()

    return u_delta


def test_serialize_deserialize():
    """
    test to_dict function for Grid Current object
    create a new grid_current object and make sure it has same properties
    """

    c_grid = GridCurrentMover(curr_file, topology_file)
    serial = c_grid.serialize('webapi')
    dict_ = GridCurrentMover.deserialize(serial)
    c2 = GridCurrentMover.new_from_dict(dict_)
    assert c_grid == c2

    c_grid.update_from_dict(dict_)  # tests no failures

