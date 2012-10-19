#!/usr/bin/env python
"""
rand.py

Contains functions for adding randomness - added 'g' for gnome random, not to confuse with standard
python random functions 
"""
import cython
import numpy
import random
from math import sqrt

def random_with_persistance(low, high, persistence, step_len):
    """
    Used by gnome to generate a randomness between low and high, which is persistent for some duration
    
    :param low: lower bound for random number
    :param high: upper bound for random number
    :param step_len: step size for the simulation
    :param persistence: in seconds. Since we add randomness for each timestep, the persistence 
    parameter is used to make the randomness invariant to step size.
    """
    if high < low:
        raise ValueError("The lower bound for random_with_persistance must be less than or equal to upper bound")

    # should an error be thrown if low < 0?
    
    if low == high:
        return low  # if they are equal, return
    
    if persistence > 0:
        orig = high - low
        range = orig * sqrt( persistence/step_len)
        mean = (high + low)/2.
        
        # update the bounds for generating the random number 
        low = mean - range/2.
        high= mean + range/2.
        
    return random.uniform(low, high)