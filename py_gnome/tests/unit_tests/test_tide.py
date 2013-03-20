'''
Test all operations for cats mover work
'''
import datetime
import os

import numpy as np
import pytest

import gnome
from gnome import environment, basic_types
from gnome.spill_container import TestSpillContainer
from gnome.utilities import time_utils

shio_file = os.path.join( os.path.dirname(__file__), r"SampleData/tides/CLISShio.txt")
ossm_file = os.path.join( os.path.dirname(__file__), r"SampleData/tides/TideHdr.FINAL")

def test_exceptions():
    """
    Test correct exceptions are raised
    """
    bad_file = r"SampleData/long_island_sound/CLISShio.txtX"
    bad_yeardata_path = r"SampleData/Data/yeardata/"
    with pytest.raises(IOError):
        environment.Tide(bad_file)
        
    with pytest.raises(IOError):
        environment.Tide(shio_file, yeardata=bad_yeardata_path)

def test_shio_file():
    """
    (WIP)
    simply tests that the file loads correctly
    """
    td = environment.Tide(shio_file)
    assert True
    
def test_ossm_file():
    """
    (WIP)
    simply tests that the file loads correctly
    """
    td = environment.Tide(ossm_file)
    assert True
    
def test_new_from_dict():
    """
    test to_dict function for Wind object
    create a new wind object and make sure it has same properties
    """
    td = environment.Tide(filename=shio_file)
    td_state = td.to_dict('create')
    td2 = environment.Tide.new_from_dict(td_state)   # this does not catch two objects with same ID
    
    assert td == td2

    
def test_from_dict():
    """
    test from_dict function for Tide object
    no values are changed so it just tests that there are no failures
    """
    wm = environment.Tide(filename=shio_file)
    wm_dict = wm.to_dict()
     
    wm.from_dict(wm_dict)
    
    for key in wm_dict.keys():
        assert wm.__getattribute__(key) == wm_dict.__getitem__(key)
    