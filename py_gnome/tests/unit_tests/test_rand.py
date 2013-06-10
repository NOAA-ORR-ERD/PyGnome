#!/usr/bin/env python
"""
Test gnome.utilities.rand.py
"""

from gnome.utilities import rand
import numpy as np
import pytest
import random

def test_exceptions():
    """
    Test exceptions
    """
    with pytest.raises(ValueError):
        rand.random_with_persistance(1,0)
        rand.random_with_persistance(1,0,0,0)
        
def test_random_with_persistance_scalar():
    """
    Since numbers are randomly generated, can only test array length 
    """
    x = rand.random_with_persistance(0,10)
    assert x >= 0 and x <= 10
    assert isinstance(x,float)
    
def test_random_with_persistance_array():
    """
    Test so the output is a numpy array of random numbers
    """
    x = rand.random_with_persistance(0,10,900, 60, 100)
    assert len(x) == 100
    assert isinstance(x,np.ndarray)
    
    
def test_random_with_persistance_low_equals_high():
    """
    if low==high, then return low - deterministic output
    """
    x = rand.random_with_persistance(10,10,900, 60)
    assert x == 10
    
    y = rand.random_with_persistance(10,10,900, 60, array_len=100)
    assert len(y) == 100
    assert np.all( y == 10)
    
    
def test_set_seed():
    """
    test set_seed to 1 works
    """
    from gnome.cy_gnome import cy_helpers
    rand.seed(1)
    
    xi = [random.uniform(0,i+1) for i in range(10)]
    ai = np.random.uniform(0,1,10)
    ci = [cy_helpers.rand() for i in range(10)]
    
    rand.seed(1)
    xf = [random.uniform(0,i+1) for i in range(10)]
    af = np.random.uniform(0,1,10)
    cf = [cy_helpers.rand() for i in range(10)]
    
    assert xi == xf
    assert np.all(ai == af)
    assert ci == cf 
