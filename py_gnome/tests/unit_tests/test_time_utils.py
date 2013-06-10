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
    assert isinstance(xn, datetime)
    x = time_utils.round_time(x, roundTo=1)
    assert isinstance(x, datetime) 
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
    
def test_time_dst():
    """
    test it works for datetime at 23 hours with daylight savings on
    test is only valid for places that have daylight savings time
    """
    x = datetime(2013, 3, 21, 23, 10)
    xn = _convert(x)
    assert np.all( x == xn)
    
    x = datetime(2013, 2, 21, 23, 10)    # no daylight savings
    xn = _convert(x)
    assert np.all( x == xn)
