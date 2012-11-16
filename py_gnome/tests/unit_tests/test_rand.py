#!/usr/bin/env python
"""
Test gnome.utilities.rand.py
"""

from gnome.utilities.rand import *
import numpy as np
import pytest

def test_exceptions():
    """
    Test exceptions
    """
    with pytest.raises(ValueError):
        random_with_persistance(1,0)
        random_with_persistance(1,0,0,0)
        
def test_random_with_persistance_scalar():
    """
    Since numbers are randomly generated, can only test array length 
    """
    x = random_with_persistance(0,10)
    assert x >= 0 and x <= 10
    assert isinstance(x,float)
    
def test_random_with_persistance_array():
    """
    Test so the output is a numpy array of random numbers
    """
    x = random_with_persistance(0,10,900, 60, 100)
    assert len(x) == 100
    assert isinstance(x,np.ndarray)
    
    
def test_random_with_persistance_low_equals_high():
    """
    if low==high, then return low - deterministic output
    """
    x = random_with_persistance(10,10,900, 60)
    assert x == 10
    
    y = random_with_persistance(10,10,900, 60, array_len=100)
    assert len(y) == 100
    assert np.all( y == 10)