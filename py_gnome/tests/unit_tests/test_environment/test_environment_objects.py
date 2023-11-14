"""
Tests for the environment objects that aren't tested elsewhere
"""
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gnome.environment.environment_objects import SteadyUniformCurrent

import gnome.scripting as gs

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output_from_tests"

# tests for SteadyUniformCurrent
def test_SteadyUniformCurrent_init():
    # has required arguments:
    with pytest.raises(TypeError):
        suc = SteadyUniformCurrent()
    with pytest.raises(TypeError):
        suc = SteadyUniformCurrent(1.0)

    suc = SteadyUniformCurrent(2.0, 180)
    assert suc.speed == 2.0
    assert suc.direction == 180.0

    suc = SteadyUniformCurrent(2.0, 180, name="Any Name")
    assert suc.name == "Any Name"


def test_SteadyUniformCurrent_repr():
    suc = SteadyUniformCurrent(2.0, 180, name="Any Name")

    rep = repr(suc)

    print(rep)

    assert rep == ("SteadyUniformCurrent(speed=2.0, direction=180, "
                   "units='m/s', name='Any Name')")


def test_SteadyUniformCurrent_set_units():
    suc = SteadyUniformCurrent(2.0, 180, units="knots")
    assert suc.speed == 1.028888
    assert suc.direction == 180.0


def test_SteadyUniformCurrent_data_start_stop():
    suc = SteadyUniformCurrent(2.0, 180)

    assert suc.data_start ==  gs.MinusInfTime
    assert suc.data_stop ==  gs.InfTime


@pytest.mark.parametrize("speed, dir, u, v",[(2.0, 0.0, 0.0, 2.0),
                                             (2.0, 45.0, math.sqrt(2.0), math.sqrt(2.0)),
                                             (2.0, 90.0, 2.0, 0.0),
                                             (2.0, 135.0, math.sqrt(2.0), -math.sqrt(2.0)),
                                             (2.0, 180.0, 0.0, -2.0),
                                             (2.0, 225.0, -math.sqrt(2.0), -math.sqrt(2.0)),
                                             (2.0, 270.0, -2.0, 0.0),
                                             (2.0, 315.0, -math.sqrt(2.0), math.sqrt(2.0)),
                                             ])
def test_SteadyUniformCurrent_uv(speed, dir, u, v):
    """
    tests that the U and V values are correct as we go around the compass
    """
    suc = SteadyUniformCurrent(speed, dir)

    assert math.isclose(suc.u, u, rel_tol=1e-15)
    assert math.isclose(suc.v, v, rel_tol=1e-15)


def test_SteadyUniformCurrent_change_speed():
    """
    tests that the U and V values are correct as we go around the compass
    """
    suc = SteadyUniformCurrent(4.6, 135.0)
    assert math.isclose(suc.u, 4.6 / math.sqrt(2), rel_tol=1e-15)
    assert math.isclose(suc.v, -4.6  / math.sqrt(2), rel_tol=1e-15)

    # change the direction
    suc.direction = 90
    assert math.isclose(suc.u, 4.6, rel_tol=1e-15)
    assert math.isclose(suc.v, 0, rel_tol=1e-15)

    # change the speed
    suc.speed = 1.0
    assert math.isclose(suc.u, 1.0, rel_tol=1e-15)
    assert math.isclose(suc.v, 0, rel_tol=1e-15)


def test_SteadyUniformCurrent_at():
    """
    tests that the U and V values are correct as we go around the compass
    """
    suc = SteadyUniformCurrent(4.6, 135.0)
    u, v = 4.6 / math.sqrt(2), -4.6 / math.sqrt(2)

    # value of points does not matter
    points = [(0, 0, 0), (0, 1, 0), (2, 3, 4), (3, 4, 0)]

    # value of time does not matter
    time = datetime.now()

    vels = suc.at(points, time)

    assert vels.shape == (4, 3)

    assert np.allclose(vels[:, 0], u)
    assert np.allclose(vels[:, 1], v)


def test_SteadyUniformCurrent_in_model():
    """
    a integration test -- can you use it in a model
    """
    start_time = "2023-08-09T12:00"
    model = gs.Model(start_time=start_time,
                     duration=gs.hours(1),
                     time_step=gs.minutes(10),
                     )
    model.spills += gs.surface_point_line_spill(num_elements=10,
                                                release_time=start_time,
                                                start_position=(0,0))
    suc = SteadyUniformCurrent(math.sqrt(2), 45, units='knots')
    c_mover = gs.CurrentMover(current=suc)

    model.movers += c_mover
    model.full_run()
    # for step in model:
    positions = model.get_spill_property('positions')
    # 1 knot for 1 hour is 1/60th of a degree
    assert np.allclose(positions[:, 0], 1 / 60)
    assert np.allclose(positions[:, 1], 1 / 60)
    assert np.allclose(positions[:, 2], 0)


def test_SteadyUniformCurrent_save_load():
    """
    test save/loading
    """
    suc = SteadyUniformCurrent(2.8, 45, units='knots')

    save_json, zipfile_, _refs = suc.save(OUTPUT_DIR)

    print(save_json)
    print(zipfile_)

    loaded = SteadyUniformCurrent.load(zipfile_)

    assert loaded == suc


def test_SteadyUniformCurrent_serialize():
    # NOTE: this is not saving the data_start and data_stop attributes
    #       they are read_only, but shouldn't they get passed to the webAPI?
    suc = SteadyUniformCurrent(2.8, 45, units='knots')

    serial = suc.to_dict(json_='webapi')

    new_suc = SteadyUniformCurrent.new_from_dict(serial)

    print(serial)

    assert suc == new_suc

