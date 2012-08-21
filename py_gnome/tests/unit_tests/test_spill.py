"""
Tests if the 
"""
import datetime

import pytest

import numpy as np

from gnome import basic_types
import gnome.spill2 as spill


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

class Test_point_release:

    def test_point_release_init(self):
        """
        can it be initialized
        """
        sp = spill.PointReleaseSpill(num_LEs = 10,
                                     start_position = (-120, 45),
                                     release_time = datetime.datetime(2012, 8, 8, 13),
                                     )
    

