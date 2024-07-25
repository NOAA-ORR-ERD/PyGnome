"""
Tests for the environment objects that aren't tested elsewhere
"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from gnome.environment.environment_objects import SteadyUniformCurrent, GridCurrent, FileGridCurrent
from gnome.spill_container import SpillContainer

import gnome.scripting as gs

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output_from_tests"
TEST_DATA_DIR = HERE / "sample_data"

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
    model.spills += gs.point_line_spill(num_elements=10,
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


def test_GridCurrent_angle():
    roms_file_degrees = TEST_DATA_DIR / "example_roms_degrees.nc"
    grid_cur_degrees = GridCurrent.from_netCDF(filename=roms_file_degrees)

    roms_file_radians = TEST_DATA_DIR / "example_roms_radians.nc"
    grid_cur_radians = GridCurrent.from_netCDF(filename=roms_file_radians)

    assert grid_cur_degrees.angle.units == "degrees"
    assert grid_cur_degrees.angle._gnome_unit == "radians"
    assert grid_cur_radians.angle.units == "radians"
    assert grid_cur_radians.angle._gnome_unit == "radians"


def test_GridCurrent_single_time():
    roms_file = TEST_DATA_DIR / "example_roms_radians.nc"
    grid_cur = GridCurrent.from_netCDF(filename=roms_file)

    assert grid_cur.extrapolation_is_allowed == True
    grid_cur.extrapolation_is_allowed = False
    # file has one time so extrapolation is always on
    assert grid_cur.extrapolation_is_allowed == True


def test_GridCurrent_at():
    """
    test getting velocity and angle from a GridCurrent
    """
    roms_file = TEST_DATA_DIR / "example_roms_radians.nc"

    rel_time = datetime(2023, 8, 9, 12, 0)
    pos = (-152.2,60.5,0)
    release = gs.PointLineRelease(rel_time,
                                   pos,
                                   num_elements=1)
    sp = gs.Spill(release=release)
    sc = SpillContainer()
    sc.spills += sp
    sc.prepare_for_model_run(array_types=sp.array_types)
    sp.release_elements(sc, rel_time, rel_time + timedelta(seconds=900))
    point = sc['positions']

    grid_cur = GridCurrent.from_netCDF(filename=roms_file)
    velocity = grid_cur.at(points=point,time=rel_time)

    u = velocity[0][0]
    v = velocity[0][1]
    w = velocity[0][2]

    assert math.isclose(u, -.58206, rel_tol=1e-5)
    assert math.isclose(v, -.99779, rel_tol=1e-5)
    assert math.isclose(w, 0.0, rel_tol=1e-5)

    angle = grid_cur.angle.at(points=point,time=rel_time)
    assert math.isclose(angle[0][0], -0.32804926, rel_tol=1e-5)


def test_GridCurrent_get_bounds():
    """
    test getting bounds from the grid
    """
    roms_file = TEST_DATA_DIR / "example_roms_radians.nc"

    grid_cur = GridCurrent.from_netCDF(filename=roms_file)
    bounds = grid_cur.get_bounds() # ((lon_min, lat_min), (lon_max, lat_max))

    assert math.isclose(bounds[0][0], -152.86, rel_tol=1e-5)
    assert math.isclose(bounds[0][1], 60.422, rel_tol=1e-5)
    assert math.isclose(bounds[1][0], -152.029, rel_tol=1e-5)
    assert math.isclose(bounds[1][1], 60.649, rel_tol=1e-5)


def test_FileGridCurrent():
    roms_file = TEST_DATA_DIR / "example_roms_two_times.nc"
    grid_cur = FileGridCurrent(filename=roms_file)
    assert grid_cur.angle.units == "radians"
    assert grid_cur.extrapolation_is_allowed == False

    grid_cur2 = FileGridCurrent(filename=roms_file,extrapolation_is_allowed=True)
    assert grid_cur2.extrapolation_is_allowed == True
