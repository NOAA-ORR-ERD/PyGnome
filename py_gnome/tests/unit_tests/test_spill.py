"""
Tests if the 
"""
import datetime

import pytest

import numpy as np

from gnome import basic_types
from gnome import spill

def test_init_simple():
    sp = spill.Spill(num_LEs = 10)
    
    assert sp['status_codes'].shape == (10,)
    assert sp['positions'].shape == (10,3)
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )


def test_data_access():
    sp = spill.Spill(num_LEs = 10)
    
    sp['positions'] += (3.0, 3.0, 3.0)

    assert np.array_equal(sp['positions'],
                          np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
                          )
    
    
def test_data_setting():
    sp = spill.Spill(num_LEs = 10)
    
    new_pos = np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
    
    sp['positions'] = new_pos

    assert np.array_equal(sp['positions'],
                          new_pos
                          )
    
def test_data_setting_error1():
    """
    Should get an error when trying to set the data to a different size array
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_pos = np.ones( (12, 3), dtype=basic_types.world_point_type ) * 3.0
    with pytest.raises(ValueError):
        sp['positions'] = new_pos


def test_data_setting_error2():
    """
    Should get an error when trying to set the data to a different type array
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_pos = np.ones( (10, 3), dtype=np.int32 )
    
    with pytest.raises(ValueError):
        sp['positions'] = new_pos

def test_data_setting_error3():
    """
    Should get an error when trying to set the data to a different shape array
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_pos = np.ones( (10, 4), dtype=basic_types.world_point_type ) * 3.0
    
    with pytest.raises(ValueError):
        sp['positions'] = new_pos

def test_data_setting_new():
    """
    Should be able to add a new data array
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_arr = np.ones( (10, 3), dtype=np.float64 )
    
    sp['new_name'] = new_arr

    assert sp['new_name'] is new_arr 

def test_data_setting_new_list():
    """
    Should be able to add a new data that's not a numpy array
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_arr = range(10)
    
    sp['new_name'] = new_arr

    assert np.array_equal(sp['new_name'],  new_arr) 
    
def test_data_setting_new_error():
    """
    Should get an error adding a new data array of the wrong size
    """
    sp = spill.Spill(num_LEs = 10)
    
    new_arr = np.ones( (12, 3), dtype=np.float64 )
    
    with pytest.raises(ValueError):
        sp['new_name'] = new_arr

## PointReleaseTests
    
def test_point_init():
    sp = spill.PointReleaseSpill(num_LEs = 10,
                                 start_position = (28.5, -128.3, 0),
                                 release_time=datetime.datetime(2012, 8, 20, 13),
                                 )
    
    assert sp['status_codes'].shape == (10,)
    assert sp['positions'].shape == (10,3)
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

def test_point_release():
    rel_time = datetime.datetime(2012, 8, 20, 13)
    sp = spill.PointReleaseSpill(num_LEs = 10,
                                 start_position = (28.5, -128.3, 0),
                                 release_time=rel_time,
                                 )
    
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

    sp.release_elements(rel_time - datetime.timedelta(seconds=1) )# one second before release time
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released )
    
    sp.release_elements(rel_time)
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )

def test_point_reset():
    rel_time = datetime.datetime(2012, 8, 20, 13)
    sp = spill.PointReleaseSpill(num_LEs = 10,
                                 start_position = (28.5, -128.3, 0),
                                 release_time=rel_time,
                                 )
    
    sp.release_elements(rel_time)
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )
    sp.reset()
    assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

    


