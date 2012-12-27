"""
Tests the SpillContainer class
"""

from datetime import datetime, timedelta

import pytest

import numpy as np

from gnome import basic_types
from gnome.spill_container import SpillContainer, TestSpillContainer
from gnome.spill import Spill, SurfaceReleaseSpill

def test_simple_init():
    sc = SpillContainer()

def test_test_spill_container():
    pos = (28.0, -75.0, 0.0)
    num_elements = 10
    sc = TestSpillContainer(num_elements, (28, -75, 0) )

    assert sc['positions'].shape == (10, 3)

    assert np.array_equal( sc['positions'][0], pos )
    assert np.array_equal( sc['positions'][-1], pos )


## real tesing involves adding spills!
def test_one_simple_spill():
    start_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements =  100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time)
    sc.add_spill(spill)
    sc.prepare_for_model_step(start_time)

    assert sc.num_elements == num_elements

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    assert np.array_equal( sc['positions'][0], start_position )

## multiple spills with different release times:
def test_multiple_spills():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements =  100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time)

    spill2 = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)
    sc.prepare_for_model_step(start_time)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    assert sc['positions'].shape == (num_elements*2, 3)
    assert sc['last_water_positions'].shape == (num_elements*2, 3)

    ## check the get_spill method
    assert sc.get_spill(spill.id) == spill
    assert sc.get_spill(spill2.id) == spill2

    ## check the remove_spill_by_id
    sc.remove_spill_by_id(spill.id)
    assert sc.get_spill(spill.id) is None # it shouldn't be there anymore.


def test_reset():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements =  100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time)

    spill2 = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)
    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    sc.reset()
    assert spill.num_released == 0
    assert spill2.num_released == 0

def test_reset2():
    """
    test that extra arrays are removed on a reset

    # not much of a test, really -- add more?
    """
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements =  100
    sc = SpillContainer()
    spill = Spill(num_elements)

    spill2 = SurfaceReleaseSpill(num_elements,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)

    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    sc.remove_spill(spill)

    sc.reset()
    print "id of spill 2", spill2.id
    assert spill2.num_released == 0



def test_data_access():
    sp = TestSpillContainer(10, (0,0,0),)
    
    sp['positions'] += (3.0, 3.0, 3.0)

    assert np.array_equal(sp['positions'],
                          np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
                          )
    
    
def test_data_setting():
    sp = TestSpillContainer(num_elements = 10 )
    
    new_pos = np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
    
    sp['positions'] = new_pos

    assert np.array_equal(sp['positions'],
                          new_pos
                          )
    
def test_data_setting_error1():
    """
    Should get an error when trying to set the data to a different size array
    """
    sp = TestSpillContainer(num_elements =  10)
    
    new_pos = np.ones( (12, 3), dtype=basic_types.world_point_type ) * 3.0
    with pytest.raises(ValueError):
        sp['positions'] = new_pos


def test_data_setting_error2():
    """
    Should get an error when trying to set the data to a different type array
    """
    sp = TestSpillContainer(num_elements =  10)
    
    new_pos = np.ones( (10, 3), dtype=np.int32 )
    
    with pytest.raises(ValueError):
        sp['positions'] = new_pos

def test_data_setting_error3():
    """
    Should get an error when trying to set the data to a different shape array
    """
    sp = TestSpillContainer(num_elements =  10)
    
    new_pos = np.ones( (10, 4), dtype=basic_types.world_point_type ) * 3.0
    
    with pytest.raises(ValueError):
        sp['positions'] = new_pos

def test_data_setting_new():
    """
    Should be able to add a new data array
    """
    sp = TestSpillContainer(num_elements =  10)
    
    new_arr = np.ones( (10, 3), dtype=np.float64 )
    
    sp['new_name'] = new_arr

    assert sp['new_name'] is new_arr 

def test_data_setting_new_list():
    """
    Should be able to add a new data that's not a numpy array
    """
    sp = TestSpillContainer(num_elements =  10)
    
    new_arr = range(10)
    
    sp['new_name'] = new_arr

    assert np.array_equal(sp['new_name'],  new_arr) 
  
## NOTE: this is no longer a feature  
# def test_data_setting_new_error():
#     """
#     Should get an error adding a new data array of the wrong size
#     """
#     sp = TestSpillContainer(num_elements =  10)
    
#     new_arr = np.ones( (12, 3), dtype=np.float64 )
    
#     with pytest.raises(ValueError):
#         sp['new_name'] = new_arr



    


