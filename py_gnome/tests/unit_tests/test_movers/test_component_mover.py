'''
Test all operations for component mover work
'''

import datetime
import os

import numpy as np
import pytest
import tempfile

from gnome.movers import ComponentMover
from gnome.environment import Wind, constant_wind
from gnome.utilities import time_utils

from ..conftest import sample_sc_release, testdata


curr1_file = testdata['ComponentMover']['curr']
curr2_file = testdata['ComponentMover']['curr2']
#wnd = Wind(filename=testdata['ComponentMover']['wind'])
wnd = constant_wind(5., 270, 'knots')


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    with pytest.raises(ValueError):
        'bad file'
        ComponentMover(os.path.join('./', 'NW30ktwinds.CURX'))


num_le = 3
start_pos = (-75.262319, 39.142987, 0)
rel_time = datetime.datetime(2012, 8, 20, 13)
time_step = 15 * 60  # seconds
model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))


# NOTE: Following expected results were obtained from Gnome for the above
#       test setup.  These are documented here, though there is no test to
#       check against these values.
# TODO: There needs to be a simpler test for testing the values.
#       Simple datafiles exist, the test just needs to be completed.
#
#         expected_shio_only = (-0.00089881, 0.00303475, 0.)
#         expected_cats_shio = ( 0.00020082,-0.00067807, 0.)
#
#       For now, the expected gnome results are documented here for above test.
# certain spill, expected results for this model_time and the above
# shio and currents file
# These were obtained from Gnome, so have been added here to explicitly
# test against.
# If any of the above setup parameters change, these results will not match!


def test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    delta = run_test_loop()

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros


# fixme -- should this be a fixture?
def run_test_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    component = ComponentMover(curr1_file,
                               curr2_file,
                               wind=wnd,
                               scale_refpoint=(-75.262319, 39.142987, 0))
    print(component.scale_refpoint)
    delta = _certain_loop(pSpill, component)

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros

    return delta


def run_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    component = ComponentMover(curr1_file, curr2_file, wind=wnd)
    component.scale_refpoint = (-75.262319, 39.142987, 0)
    u_delta = _uncertain_loop(pSpill, component)

    _assert_move(u_delta)

    return u_delta


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


def test_get_scaled_velocities_two_patterns():
    component = ComponentMover(curr1_file,
                               curr2_file,
                               wind=wnd,
                               scale_refpoint=(-75.262319, 39.142987, 0))
    vels = component.get_scaled_velocities(time_utils.date_to_sec(rel_time))

    # not much of test -- but at least we're not getting NaNs...
    assert np.alltrue(np.isfinite(vels['u']))
    assert np.alltrue(np.isfinite(vels['v']))

    # note: these values were pulled from making this call
    #       so may not be correct, but this will catch a regression
    assert vels['u'][1] == -0.23194323751895507
    assert vels['v'][10] == -0.09848350841987873
    assert vels['u'][1] == -0.23194323751895507
    assert vels['v'][10] == -0.09848350841987873

def test_get_scaled_velocities_one_pattern():
    component = ComponentMover(filename1=curr1_file,
                               wind=wnd,
                               scale_refpoint=(-75.262319, 39.142987, 0))
    vels = component.get_scaled_velocities(time_utils.date_to_sec(rel_time))

    # not much of test -- but at least we're not getting NaNs...
    assert np.alltrue(np.isfinite(vels['u']))
    assert np.alltrue(np.isfinite(vels['v']))

    # note: these values were pulled from making this call
    #       so may not be correct, but this will catch a regression
    assert vels['u'][1] == 2.34578240814883e-16
    assert vels['v'][10] == 7.692065313287368e-16
    assert vels['u'][1] == 2.34578240814883e-16
    assert vels['v'][10] == 7.692065313287368e-16



# set one up with only one pattern for the next suite of tests
c_component = ComponentMover(curr1_file)


def test_default_props():
    """
    test default properties
    """

    assert c_component.pat1_angle == 0
    assert c_component.pat1_speed == 10
    #assert c_component.scale_refpoint is None


def test_pat1_angle():
    """
    test setting / getting properties
    """

    c_component.pat1_angle = 20
    assert c_component.pat1_angle == 20


def test_pat1_speed():
    """
    test setting / getting properties
    """

    c_component.pat1_speed = 5
    print(c_component.pat1_speed)
    assert c_component.pat1_speed == 5



@pytest.mark.parametrize("tgt", [(1, 2, 3), (5, 6)])
def test_scale_refpoint(tgt):
    """
    test setting / getting properties
    """

    if len(tgt) == 2:
        exp_tgt = (tgt[0], tgt[1], 0.)
    else:
        exp_tgt = tgt
    c_component.scale_refpoint = tgt  # can be a list or a tuple
    assert c_component.scale_refpoint == tuple(exp_tgt)
    c_component.scale_refpoint = list(tgt)  # can be a list or a tuple
    assert c_component.scale_refpoint == tuple(exp_tgt)


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print()
    print(delta)
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, component):
    component.prepare_for_model_run()
    component.prepare_for_model_step(pSpill, time_step, model_time)
    delta = component.get_move(pSpill, time_step, model_time)
    component.model_step_is_done()

    return delta


def _uncertain_loop(pSpill, component):
    component.prepare_for_model_run()
    component.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = component.get_move(pSpill, time_step, model_time)
    component.model_step_is_done()

    return u_delta


@pytest.mark.parametrize("wind", (None, wnd))
def test_serialize_deserialize(wind):
    """
    test to_dict function for Component mover with wind object
    create a new Component mover and make sure it has same properties
    """

    c_component = ComponentMover(curr1_file, wind=wind)
    serial = c_component.serialize()
    deser =  ComponentMover.deserialize(serial)

    assert deser == c_component

@pytest.mark.parametrize("wind", (None, wnd))
def test_save_load(wind):
    """
    test to_dict function for Component mover with wind object
    create a new Component mover and make sure it has same properties
    """

    saveloc = tempfile.mkdtemp()
    c_component = ComponentMover(curr1_file, wind=wind)
    json_, saveloc, refs = c_component.save(saveloc)
    loaded =  ComponentMover.load(saveloc)

    assert loaded == c_component

