#!/usr/bin/env python
"""
rand.py

Contains functions for adding randomness - added 'g' for gnome random, not to confuse with standard
python random functions 
"""
import cython
import numpy as np
import random
from math import sqrt

def random_with_persistance(low, high, persistence=0, time_step=1, array_len=1):
    """
    Used by gnome to generate a randomness between low and high, which is persistent for duration time_step
    
    :param low: lower bound for random number
    :param high: upper bound for random number
    :param time_step: step size for the simulation in seconds.
    :param persistence: in seconds. Since we add randomness for each timestep, the persistence 
    parameter is used to make the randomness invariant to size of time_step.
    
    Note: persistence and time_step should be in the same units
    """
    if high < low:
        raise ValueError("The lower bound for random_with_persistance must be less than or equal to upper bound")

    if array_len < 1:
        raise ValueError("array_len must be >= 1. This is the length of the array containing randomly generated numbers")

    # should an error be thrown if low < 0?    
    if low == high:
        if array_len == 1:
            return low
        else:
            x = np.ndarray((array_len,))
            x[:] = low
            return x
    
    if persistence > 0:
        orig = high - low
        range = orig * sqrt( persistence/time_step)
        mean = (high + low)/2.
        
        # update the bounds for generating the random number 
        low = mean - range/2.
        high= mean + range/2.
     
    if array_len == 1:   
        return np.random.uniform(low, high)
    else:
        return np.random.uniform(low,high,array_len)