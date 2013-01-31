'''
Test all operations for cats mover work
'''
import gnome
from gnome import movers, basic_types
from gnome.spill_container import TestSpillContainer
from gnome.utilities import time_utils
import datetime

import numpy as np
import pytest

shio_file = r"SampleData/long_island_sound/CLISShio.txt"
curr_file=r"SampleData/long_island_sound/tidesWAC.CUR"

def test_exceptions():
    """
    Test correct exceptions are raised
    """
    bad_file=r"SampleData/long_island_sound/tidesWAC.CURX"
    bad_shio_file = r"SampleData/long_island_sound/CLISShio.txtX"
    with pytest.raises(ValueError):
        movers.CatsMover(bad_file)
        movers.CatsMover(curr_file, bad_shio_file)


num_le = 3
start_pos = (-72.5, 41.17, 0)
rel_time = datetime.datetime(2012, 8, 20, 13)
time_step = 15*60 # seconds

model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)

def test_loop():
    """
    test one time step with no uncertainty on the spill
    """
    pSpill = TestSpillContainer(num_le, start_pos, rel_time)
    cats = movers.CatsMover(curr_file, shio_file)
    cats.prepare_for_model_run()
    cats.prepare_for_model_step(pSpill, time_step, model_time)
    delta = cats.get_move(pSpill, time_step, model_time)
    cats.model_step_is_done()
    
    _assert_move(delta)
    assert np.all( delta[:,0]== delta[0,0] )    # lat move is the same for all LEs
    assert np.all( delta[:,1]== delta[0,1] )    # long move is the same for all LEs
    assert np.all( delta[:,2] == 0 )    # 'z' is zeros
    return delta
    

def test_uncertain_loop():
    """
    test one time step with uncertainty on the spill
    """
    pSpill = TestSpillContainer(num_le, start_pos, rel_time, uncertain=True) 
    cats = movers.CatsMover(curr_file, shio_file)
    cats.prepare_for_model_run()
    cats.prepare_for_model_step(pSpill, time_step, model_time)
    delta = cats.get_move(pSpill, time_step, model_time)
    cats.model_step_is_done()
    
    _assert_move(delta)
    return delta

def test_certain_uncertain():
    """
    make sure certain and uncertain loops are different
    """
    delta  = test_loop()
    u_delta= test_uncertain_loop()
    assert np.all(delta[:,:2] != u_delta[:,:2])
    assert np.all(delta[:,2] == u_delta[:,2])

def _assert_move(delta):
    """
    helper function to test assertions
    """
    print
    print delta
    assert np.all( delta[:,:2] != 0)
    assert np.all( delta[:,2] == 0)
