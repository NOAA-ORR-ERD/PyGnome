'''
Test all operations for py_current mover work
'''

import datetime

import numpy as np
import pytest
import tempfile
import zipfile

from gnome.movers import CurrentMover
from gnome.environment.environment_objects import GridCurrent, SteadyUniformCurrent
from gnome.utilities import time_utils
import gnome.scripting as gs

from ..conftest import (sample_sc_release, testdata, validate_serialize_json,
                        validate_save_json)

curr_file = testdata['c_GridCurrentMover'][
    'curr_reg']  #just a regular grid netcdf - this fails save_load
curr_file2 = testdata['c_GridCurrentMover'][
    'curr_tri']  #a triangular grid netcdf
file_list = [testdata['c_GridCurrentMover'][
    'series_file1'],
	testdata['c_GridCurrentMover'][
    'series_file2'],
    ]


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    #bad_file = os.path.join('./', 'tidesWAC.CURX')
    #bad_file = None
    with pytest.raises(ValueError):
        CurrentMover()


num_le = 10
start_pos = (3.549, 51.88, 0)
rel_time = datetime.datetime(1999, 11, 29, 21)
#rel_time = datetime.datetime(2004, 12, 31, 13) # date for curr_file2
time_step = 15 * 60  # seconds
model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))


def run_test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    current = GridCurrent.from_netCDF(curr_file)
    py_current = CurrentMover(current=current)
    delta = _certain_loop(pSpill, py_current)

    return delta


def test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    current = GridCurrent.from_netCDF(curr_file)
    py_current = CurrentMover(current=current)
    delta = run_test_loop()

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros

    #return delta


def run_uncertain_loop():
    """
    runs one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time, uncertain=True)
    current = GridCurrent.from_netCDF(curr_file)
    py_current = CurrentMover(current=current)
    u_delta = _uncertain_loop(pSpill, py_current)

    _assert_move(u_delta)

    return u_delta


def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time, uncertain=True)
    current = GridCurrent.from_netCDF(curr_file)
    py_current = CurrentMover(current=current)
    u_delta = run_uncertain_loop()

    _assert_move(u_delta)


def test_certain_uncertain():
    """
    make sure certain and uncertain loop results in different deltas
    """

    delta = run_test_loop()
    u_delta = run_uncertain_loop()
    print()
    print(delta)
    print(u_delta)
    assert np.all(delta[:, :2] != u_delta[:, :2])
    assert np.all(delta[:, 2] == u_delta[:, 2])


def test_default_props():
    """
    test default properties
    """
    current = GridCurrent.from_netCDF(filename=curr_file)
    py_cur = CurrentMover(current=current)

    assert py_cur.uncertain_duration == 24 * 3600
    assert py_cur.uncertain_time_delay == 0
    assert py_cur.uncertain_along == 0.5
    assert py_cur.uncertain_cross == 0.25
    assert py_cur.scale_value == 1
    #assert py_cur.time_offset == 0
    assert py_cur.default_num_method == 'RK2'
    #assert py_cur.grid_topology == None


def test_non_default_props():
    """
    test default properties
    """
    current = GridCurrent.from_netCDF(filename=curr_file)
    py_cur = CurrentMover(current=current,
                          uncertain_duration=2 * 3600,
                          uncertain_time_delay=12 * 3600,
                          uncertain_along=0.3,
                          uncertain_cross=0.15,
                          scale_value=1.2,
                          default_num_method='Euler')

    assert py_cur.uncertain_duration == 2 * 3600
    assert py_cur.uncertain_time_delay == 12 * 3600
    assert py_cur.uncertain_along == 0.3
    assert py_cur.uncertain_cross == 0.15
    assert py_cur.scale_value == 1.2
    assert py_cur.default_num_method == 'Euler'


def test_update_uncertainty():
    """
    this is doing a fair bit, but need to test somehow
    """
    curr = SteadyUniformCurrent(speed=1, direction=45, units='m/s')
    mov = CurrentMover(
        current=curr,
        uncertain_duration=24 * 3600,
        uncertain_time_delay=0,
        uncertain_along=.5,
        uncertain_cross=.25,
    )

    print(mov._uncertainty_list)
    # should start out empty
    assert len(mov._uncertainty_list) == 0

    mov._update_uncertainty(
        num_les=50, elapsed_time=3600)  # time in seconds -- should it be?
    # should be 50 now
    assert len(mov._uncertainty_list) == 50

    mov._update_uncertainty(
        num_les=75, elapsed_time=3600)  # time in seconds -- should it be?
    # should be 75 now
    assert len(mov._uncertainty_list) == 75

    uncertainty_list = mov._uncertainty_list
    assert np.all((uncertainty_list[:, 0] >= -0.5)
                  & (uncertainty_list[:, 0] <= 0.5))
    assert np.all((uncertainty_list[:, 1] >= -0.25)
                  & (uncertainty_list[:, 1] <= 0.25))

    # what else to check?


def test_add_uncertainty():
    curr = SteadyUniformCurrent(speed=1, direction=45, units='m/s')
    # note: current isn't used for this test, but we need one to create the mover
    mov = CurrentMover(current=curr)

    # should be the same length as the deltas
    mov._uncertainty_list = np.array([
        (.2, .1),
        (.2, .1),
        (.2, .1),
        (.2, .1),
        (.2, .1),
        (.2, .1),
        (.2, .1),
        (.2, .1),
    ],
                                     dtype=np.float64)
    # some deltas to test (units? hopefully meters)
    deltas = np.array([
        (10, 0, 0),
        (0, 10, 0),
        (-10, 0, 0),
        (0, -10, 0),
        (10, 10, 0),
        (10, -10, 0),
        (-10, -10, 0),
        (-10, 10, 0),
    ],
                      dtype=np.float64)

    new_deltas = mov._add_uncertainty(deltas)

    print(repr(new_deltas))

    # NOTE: this seems to be rotating left or right, depending on the
    #       quadrant -- which is probably OK, given that the uncertainty
    #       should be symmetric.

    correct = np.array([
        [12., -1.2, 0.],
        [1., 11.9, 0.],
        [-12., 1.2, 0.],
        [-1., -11.9, 0.],
        [13., 10.7, 0.],
        [11., -13.1, 0.],
        [-13., -10.7, 0.],
        [-11., 13.1, 0.],
    ],
                       dtype=np.float64)

    assert np.allclose(new_deltas, correct)


def test_uncertainty_deltas():
    """
    test if the actual deltas resulting from uncertainty is correct
    """
    curr = SteadyUniformCurrent(speed=1.0, direction=0, units='m/s')
    mov = CurrentMover(current=curr)

    sc = sample_sc_release(
        num_elements=2,
        start_pos=(-78.0, 48.0, 0.0),
        release_time=datetime.datetime(2000, 1, 1, 1),
        uncertain=True,
    )

    # hack to reset the positions -- so we can test deltas at different latitudes
    sc['positions'] = np.array([
        [-78., 0., 0.],
        [-78., 45., 0.],
    ])

    # should be the same length as the deltas
    mov._uncertainty_list = np.array(
        [
            (.2, .1),
            (.2, .1),
        ],
        dtype=np.float64)

    deltas = mov.get_move(
        sc,
        time_step=100,
        model_time_datetime=datetime.datetime.now(),
        num_method='Euler',
    )

    # the deltas should be the same for both points, when adjusted for latitude:

    lat = sc['positions'][:, 1]
    # approx conversion to meters
    deltas[:, 0] = deltas[:, 0] * np.cos(np.deg2rad(lat))
    deltas *= 111120

    # think this is right, but I'm still a bit confused about the details
    # D = 100 * 1  # 1 m/s * 100s
    # correct = (D * 0.1, D * 1.2, 0) # not quite right
    correct = (10., 119., 0.)  # from the code

    assert np.all(deltas[0, :] == deltas[1, :])
    assert np.allclose(deltas, correct, rtol=1e-8)


def test_file_list():
    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    current = GridCurrent.from_netCDF(file_list)
    py_current = CurrentMover(current=current)
    delta = _certain_loop(pSpill, py_current)

    _assert_move(delta)

    new_time = datetime.datetime(1999, 11, 30, 21) # second file

    py_current.prepare_for_model_step(pSpill, time_step, new_time)
    delta = py_current.get_move(pSpill, time_step, new_time)
    py_current.model_step_is_done(pSpill)

    _assert_move(delta)


def test_scale_value():
    """
    test setting / getting properties

    but does it work?
    """
    current = GridCurrent.from_netCDF(filename=curr_file)
    py_cur = CurrentMover(current=current)

    py_cur.scale_value = 0
    print(py_cur.scale_value)
    assert py_cur.scale_value == 0


# Helper functions for tests


def _assert_move(delta):
    """
    helper function to test assertions
    """

    print()
    print(delta)
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, py_current):
    py_current.prepare_for_model_run()
    py_current.prepare_for_model_step(pSpill, time_step, model_time)
    delta = py_current.get_move(pSpill, time_step, model_time)
    py_current.model_step_is_done(pSpill)

    return delta


def _uncertain_loop(pSpill, py_current):
    py_current.prepare_for_model_run()
    py_current.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = py_current.get_move(pSpill, time_step, model_time)
    py_current.model_step_is_done(pSpill)

    return u_delta


def test_serialize_deserialize():
    """
    test serialize/deserialize/update_from_dict doesn't raise errors
    """
    current = GridCurrent.from_netCDF(curr_file2)
    py_current = CurrentMover(current=current)

    serial = py_current.serialize()
    assert validate_serialize_json(serial, py_current)

    # check our CurrentMover attributes

    deser = CurrentMover.deserialize(serial)

    assert deser == py_current


def test_save_load():
    """
    test save/loading
    """

    saveloc = tempfile.mkdtemp()
    current = GridCurrent.from_netCDF(curr_file2)
    py_current = CurrentMover(current=current)
    save_json, zipfile_, _refs = py_current.save(saveloc)

    assert validate_save_json(save_json, zipfile.ZipFile(zipfile_), py_current)

    loaded = CurrentMover.load(zipfile_)

    assert loaded == py_current
