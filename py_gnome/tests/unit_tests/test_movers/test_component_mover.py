'''
Test all operations for component mover work
'''

import datetime
import os

import numpy as np
import pytest

from gnome.movers import ComponentMover
from gnome.environment import Wind
from gnome.utilities import time_utils
from gnome.utilities.remote_data import get_datafile
from gnome.persist import load

from conftest import sample_sc_release

here = os.path.dirname(__file__)
lis_dir = os.path.join(here, 'sample_data', 'delaware_bay')

curr1_file = get_datafile(os.path.join(lis_dir, 'NW30ktwinds.cur'))
curr2_file = get_datafile(os.path.join(lis_dir, 'SW30ktwinds.cur'))
wnd = Wind(filename=get_datafile(os.path.join(lis_dir, 'ConstantWind.WND')))


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    bad_file = os.path.join(lis_dir, 'NW30ktwinds.CURX')
    with pytest.raises(ValueError):
        ComponentMover(bad_file)

    with pytest.raises(TypeError):
        ComponentMover(curr1_file, wind=10)


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

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    component = ComponentMover(curr1_file, curr2_file, wind=wnd, ref_point = (-75.262319, 39.142987))
    component.ref_point = (-75.262319, 39.142987)
    delta = _certain_loop(pSpill, component)

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
    component = ComponentMover(curr1_file, curr2_file, wind=wnd)
    component.ref_point = (-75.262319, 39.142987)
    u_delta = _uncertain_loop(pSpill, component)

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
    print c_component.pat1_speed
    assert c_component.pat1_speed == 5


def test_ref_point():
    """
    test setting / getting properties
    """

    tgt = (1, 2)
    c_component.ref_point = tgt  # can be a list or a tuple
    assert c_component.ref_point == tuple(tgt)
    c_component.ref_point = list(tgt)  # can be a list or a tuple
    assert c_component.ref_point == tuple(tgt)


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print
    print delta
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
    serial = c_component.serialize('webapi')
    dict_ = c_component.deserialize(serial)
    if wind:
        #assert serial['wind'] == wnd.serialize(json_)
        assert 'wind' in serial
        dict_['wind'] = wnd  # no longer updating properties of nested objects
        assert c_component.wind is wnd
    else:
        c_component.update_from_dict(dict_)
