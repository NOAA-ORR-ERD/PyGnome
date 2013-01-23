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



#
# TODO: Something is not right here - need to dig deeper in CATS - this test sometimes causes test_cy_cats_mover.TestCats.certain_move assertions to fail ..?   
#
#===============================================================================
# num_le = 3
# start_pos = (-72.5, 41.17, 0)
# rel_time = datetime.datetime(2012, 8, 20, 13)
# time_step = 15*60 # seconds
# pSpill = TestSpillContainer(num_le, start_pos, rel_time)
# 
# model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
#    
# 
# def test_loop():
#    cats = movers.CatsMover(curr_file, shio_file)
#    cats.prepare_for_model_run()
#    cats.prepare_for_model_step(pSpill, time_step, model_time)
#    delta = cats.get_move(pSpill, time_step, model_time)
#    cats.model_step_is_done()
#    print delta    
#    #assert np.all( delta != 0 )
#    
#    
# test_loop()
#===============================================================================

#===============================================================================
# def test_uncertain_loop():
#    pSpill.uncertain = True 
#    cats.prepare_for_model_run()
#    cats.prepare_for_model_step(pSpill, time_step, model_time)
#    delta = cats.get_move(pSpill, time_step, model_time)
#    cats.model_step_is_done()
#    assert np.all( delta != 0 )
#===============================================================================
    