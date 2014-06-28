'''
Test all operations for gridcurrent mover work
'''

import datetime
import os

import numpy as np
import pytest

from gnome.movers import GridWindMover
from gnome.utilities import time_utils
from gnome.utilities.remote_data import get_datafile
from gnome.persist import load

from conftest import sample_sc_release
# default settings are the same for both objects
from test_wind_mover import _defaults

here = os.path.dirname(__file__)
wind_dir = os.path.join(here, 'sample_data', 'winds')

wind_file = get_datafile(os.path.join(wind_dir, 'WindSpeedDirSubset.nc'))
topology_file = get_datafile(os.path.join(wind_dir,
                                          'WindSpeedDirSubsetTop.dat'))


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    with pytest.raises(TypeError):
        GridWindMover()

    bad_file = os.path.join(wind_dir, 'WindSpeedDirSubset.CUR')
    with pytest.raises(ValueError):
        GridWindMover(bad_file)

    with pytest.raises(TypeError):
        GridWindMover(wind_file, topology_file=10)

    with pytest.raises(ValueError):
        # todo: Following fails - cannot raise exception during initialize
        # Need to look into this issue
        #gw = GridWindMover(grid_file, uncertain_angle_units='xyz')
        gw = GridWindMover(wind_file,topology_file)   # todo: why does this fail
        gw.set_uncertain_angle(.4, 'xyz')


def test_string_repr_no_errors():
    gw = GridWindMover(wind_file,topology_file)
    print
    print '======================'
    print 'repr(WindMover): '
    print repr(gw)
    print
    print 'str(WindMover): '
    print str(gw)
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
    wind = GridWindMover(wind_file, topology_file)
    delta = _certain_loop(pSpill, wind)

    _assert_move(delta)

    #set windage to be constant or each particle has a different position, doesn't work with uncertainty on
    #assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    #assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs

    # returned delta is used in test_certain_uncertain test
    return delta


def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = GridWindMover(wind_file, topology_file)
    u_delta = _uncertain_loop(pSpill, wind)

    _assert_move(u_delta)

    # returned delta is used in test_certain_uncertain test
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


w_grid = GridWindMover(wind_file, topology_file)


def test_default_props():
    """
    test default properties
    use _defaults helper function defined in test_wind_mover.py
    """
    #==========================================================================
    # assert w_grid.active == True  # timespan is as big as possible
    # assert w_grid.uncertain_duration == 3.0
    # assert w_grid.uncertain_time_delay == 0
    # assert w_grid.uncertain_speed_scale == 2
    # assert w_grid.uncertain_angle_scale == 0.4
    # assert w_grid.uncertain_angle_units == 'rad'
    #==========================================================================
    assert w_grid.wind_scale == 1
    assert w_grid.extrapolate == False
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

    print
    print delta
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
    test to_dict function for GridWind object
    create a new grid_wind object and make sure it has same properties
    """

    grid_wind = GridWindMover(wind_file, topology_file)
    serial = grid_wind.serialize('webapi')
    dict_ = GridWindMover.deserialize(serial)
    gw2 = GridWindMover.new_from_dict(dict_)

    assert grid_wind == gw2

    grid_wind.update_from_dict(dict_)
