'''
Test all operations for py_wind mover work
'''




import datetime
import os
from os.path import basename

import numpy as np
import pytest
import tempfile
import zipfile

from gnome.movers import WindMover
from gnome.environment.environment_objects import GridWind
from gnome.utilities import time_utils

from ..conftest import (sample_sc_release,
                        testdata,
                        validate_serialize_json,
                        validate_save_json)


wind_file = testdata['c_GridWindMover']['wind_rect'] #just a regular grid netcdf 


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    # needs a file or a wind
    with pytest.raises(ValueError):
        WindMover()


num_le = 10
start_pos = (3.549, 51.88, 0)
rel_time = datetime.datetime(1999, 11, 29, 21)
time_step = 15 * 60  # seconds
model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))


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

    pSpill = sample_sc_release(num_le, start_pos, rel_time, windage_range=(0.01, 0.01))
    wind = GridWind.from_netCDF(wind_file)
    py_wind = WindMover(wind=wind)
    delta = _certain_loop(pSpill, py_wind)

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros

    return delta


def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    wind = GridWind.from_netCDF(wind_file)
    py_wind = WindMover(wind=wind)
    u_delta = _uncertain_loop(pSpill, py_wind)

    _assert_move(u_delta)

    return u_delta


def test_certain_uncertain():
    """
    make sure certain and uncertain loop results in different deltas
    """

    delta = test_loop()
    u_delta = test_uncertain_loop()
    print()
    print(delta)
    print(u_delta)
    assert np.all(delta[:, :2] != u_delta[:, :2])
    assert np.all(delta[:, 2] == u_delta[:, 2])


wind = GridWind.from_netCDF(wind_file)
py_wind = WindMover(wind=wind)


def test_default_props():
    """
    test default properties
    """
    assert py_wind.uncertain_duration == 3 * 3600
    assert py_wind.uncertain_time_delay == 0
    assert py_wind.uncertain_speed_scale == 2.
    assert py_wind.uncertain_angle_scale == 0.4
    assert py_wind.scale_value == 1
    #assert pywindr.time_offset == 0
    assert py_wind.default_num_method == 'RK2'
    #assert py_wind.grid_topology == None


def test_scale_value():
    """
    test setting / getting properties
    """

    py_wind.scale_value = 0
    print(py_wind.scale_value)
    assert py_wind.scale_value == 0


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print()
    print(delta)
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, py_wind):
    py_wind.prepare_for_model_run()
    py_wind.prepare_for_model_step(pSpill, time_step, model_time)
    delta = py_wind.get_move(pSpill, time_step, model_time)
    py_wind.model_step_is_done(pSpill)

    return delta


def _uncertain_loop(pSpill, py_wind):
    py_wind.prepare_for_model_run()
    py_wind.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = py_wind.get_move(pSpill, time_step, model_time)
    py_wind.model_step_is_done(pSpill)

    return u_delta


@pytest.mark.skip("these are not working")
def test_serialize_deserialize():
    """
    test serialize/deserialize/update_from_dict doesn't raise errors
    """
    wind = GridWind.from_netCDF(wind_file)
    py_wind = WindMover(wind=wind)

    serial = py_wind.serialize()
    assert validate_serialize_json(serial, py_wind)

    # check our WindMover attributes

    deser = WindMover.deserialize(serial)

    assert deser == py_wind


@pytest.mark.skip("these are not working")
def test_save_load():
    """
    test save/loading
    """

    saveloc = tempfile.mkdtemp()
    wind = GridWind.from_netCDF(wind_file)
    py_wind = WindMover(wind=wind)
    save_json, zipfile_, _refs = py_wind.save(saveloc)

    assert validate_save_json(save_json, zipfile.ZipFile(zipfile_), py_wind)

    loaded = WindMover.load(zipfile_)

    assert loaded == py_wind
