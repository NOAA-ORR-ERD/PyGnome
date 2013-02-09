#!/usr/bin/evn python
"""
test time_utils different input formats
"""
from datetime import datetime
import numpy as np
from gnome.utilities import time_utils

def _convert(x):
    """
    helper method for the next 4 tests
    """
    y = time_utils.date_to_sec(x)
    xn= time_utils.sec_to_date(y)
    return xn

def test_scalar_input():
    """
    test time_utils conversion return a scalar if that is what the user input
    
    always returns a numpy object
    """
    x = datetime.now()
    xn = _convert(x)
    assert type(xn) == datetime
    x = time_utils.round_time(x, roundTo=1)
    assert type(x) == datetime 
    assert x == xn
    
def test_datetime_array():
    """
    test time_utils conversion works for python datetime object
    """
    x = np.zeros((3,), dtype=datetime)
    xn = _convert(x)
    assert np.all( time_utils.round_time(x, roundTo=1) == xn)

def test_numpy_array():
    """
    time_utils works for numpy datetime object
    """
    x = np.zeros((3,), dtype='datetime64[s]')
    xn = _convert(x)
    assert np.all( x == xn)