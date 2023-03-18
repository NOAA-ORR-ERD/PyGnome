'''
Test all operations for cats mover work
'''




import datetime
import os
from os.path import basename

import numpy as np
import pytest
import tempfile
import zipfile

from gnome.movers import CatsMover
from gnome.environment import Tide
from gnome.utilities import time_utils

from ..conftest import (sample_sc_release,
                        testdata,
                        validate_serialize_json,
                        validate_save_json)


curr_file = testdata['CatsMover']['curr']
td = Tide(filename=testdata['CatsMover']['tide'])


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    bad_file = os.path.join('./', 'tidesWAC.CURX')
    with pytest.raises(ValueError):
        CatsMover(bad_file)

    with pytest.raises(TypeError):
        CatsMover(curr_file, tide=10)


num_le = 3
start_pos = (-72.5, 41.17, 0)
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
    cats = CatsMover(curr_file, tide=td)
    delta = _certain_loop(pSpill, cats)

    _assert_move(delta)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # long move matches for all LEs
    assert np.all(delta[:, 2] == 0)  # 'z' is zeros

def run_loop():
    """
    test one time step with no uncertainty on the spill
    checks there is non-zero motion.
    also checks the motion is same for all LEs
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time)
    cats = CatsMover(curr_file, tide=td)
    delta = _certain_loop(pSpill, cats)

    return delta

def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    cats = CatsMover(curr_file, tide=td)
    u_delta = _uncertain_loop(pSpill, cats)

    _assert_move(u_delta)

def run_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    checks there is non-zero motion.
    """

    pSpill = sample_sc_release(num_le, start_pos, rel_time,
                               uncertain=True)
    cats = CatsMover(curr_file, tide=td)
    u_delta = _uncertain_loop(pSpill, cats)

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


c_cats = CatsMover(curr_file)

0
def test_default_props():
    """
    test default properties
    """
    assert c_cats.uncertain_duration == 48.0
    assert c_cats.uncertain_time_delay == 0
    assert c_cats.up_cur_uncertain == 0.3
    assert c_cats.down_cur_uncertain == -0.3
    assert c_cats.right_cur_uncertain == 0.1
    assert c_cats.left_cur_uncertain == -0.1
    assert c_cats.scale is False
    assert c_cats.scale_value == 1
    assert c_cats.scale_refpoint is None


def test_scale_value():
    """
    test setting / getting properties
    """

    c_cats.scale_value = 0
    print(c_cats.scale_value)
    assert c_cats.scale_value == 0


@pytest.mark.parametrize("tgt", [(1, 2, 3), (5, 6)])
def test_scale_refpoint(tgt):
    """
    test setting / getting properties
    """

    c_cats.scale_refpoint = tgt  # can be a list or a tuple
    if len(tgt) == 2:
        exp_tgt = (tgt[0], tgt[1], 0.)
    else:
        exp_tgt = tgt

    assert c_cats.scale_refpoint == tuple(exp_tgt)
    c_cats.scale_refpoint = list(tgt)  # can be a list or a tuple
    assert c_cats.scale_refpoint == tuple(exp_tgt)


# Helper functions for tests

def _assert_move(delta):
    """
    helper function to test assertions
    """

    print()
    print(delta)
    assert np.all(delta[:, :2] != 0)
    assert np.all(delta[:, 2] == 0)


def _certain_loop(pSpill, cats):
    cats.prepare_for_model_run()
    cats.prepare_for_model_step(pSpill, time_step, model_time)
    delta = cats.get_move(pSpill, time_step, model_time)
    cats.model_step_is_done()

    return delta


def _uncertain_loop(pSpill, cats):
    cats.prepare_for_model_run()
    cats.prepare_for_model_step(pSpill, time_step, model_time)
    u_delta = cats.get_move(pSpill, time_step, model_time)
    cats.model_step_is_done()

    return u_delta


@pytest.mark.parametrize(('tide'),
                         [None, td])
def test_serialize_deserialize(tide):
    """
    test serialize/deserialize/update_from_dict doesn't raise errors
    """
    c_cats = CatsMover(curr_file, tide=tide)

    serial = c_cats.serialize()
    assert validate_serialize_json(serial, c_cats)

    # check our CatsMover attributes

    deser = CatsMover.deserialize(serial)

    assert deser == c_cats


@pytest.mark.parametrize(('tide'), [None, td])
def test_save_load(tide):
    """
    test save/loading with and without tide
    """

    saveloc = tempfile.mkdtemp()
    c_cats = CatsMover(curr_file, tide=tide)
    save_json, zipfile_, _refs = c_cats.save(saveloc)

    assert validate_save_json(save_json, zipfile.ZipFile(zipfile_), c_cats)

    loaded = CatsMover.load(zipfile_)

    assert loaded == c_cats
