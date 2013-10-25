"""
rand.py

Contains functions for adding randomness - added 'g' for gnome random, not to confuse with standard
python random functions 
"""

import numpy as np
from math import sqrt
from gnome.cy_gnome import cy_helpers
import random


##fixme: change this to take the windage array as input parameter, then change in place

def random_with_persistance(
    low,
    high,
    persistence=None,
    time_step=1.,
    array_len=None,
    ):
    """
    Used by gnome to generate a randomness between low and high, which is
    persistent for duration time_step

    :param low: lower bound for random number - could be an array, tuple, list
    :param high: upper bound for random number - could be an array, tuple, list
    :param time_step: step size for the simulation in seconds.
    :param persistence: in seconds. Since we add randomness for each timestep,
        the persistence parameter is used to make the randomness invariant to
        size of time_step. Default is None. If persistence is None, it gets set
        to either 0 or a an array of 0s of len(high)

    Note: persistence and time_step should be in the same units
          Assumes both low and high are int/float or arrays, lists, tuples.
          For lists, arrays, tuples - it converts input to numpy array with
          dtype=float

          If 'low' is an array, list or tuple, then 'high' and 'persistence'
          should also be the same length array, list or tuple. Give
          all 3 parameters for each element of the array.
    """

    msg = ('The lower bound for random_with_persistance must be less than or'
           ' equal to upper bound')

    if isinstance(low, int) or isinstance(low, float):
        # this is only used if high and low are scalars
        # would be better to have common code for persistence > 0 but
        # this seemed like the easiest way
        inp_isarray = False
        if (high < low):
            raise ValueError(msg)
        if low == high:
            return low

        if persistence is not None and persistence > 0:
            orig = high - low
            l__range = orig * sqrt(persistence / float(time_step))
            mean = (high + low) / 2.

            # update the bounds for generating the random number
            low = mean - l__range / 2.
            high = mean + l__range / 2.

    else:
        inp_isarray = True

        arr_len_msg = ("Length of 'low', 'high' and 'persistence' arrays"
                       " should be equal")
        if (len(high) != len(low)):
            raise ValueError(arr_len_msg)

        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        if np.all(high < low):
            raise ValueError(msg)

        if np.all(low == high):
            return low  # ignore array_len parameter

        if persistence is not None:
            persistence = np.asarray(persistence, dtype=float)
            if (len(high) != len(persistence)):
                raise ValueError(arr_len_msg)

            mask = (persistence > 0)

            if np.any(mask):
                orig = high[mask] - low[mask]
                l__range = orig * np.sqrt(persistence[mask] / float(time_step))
                mean = (high[mask] + low[mask]) / 2.

                # update the bounds for generating the random number
                low[mask] = mean - l__range / 2.
                high[mask] = mean + l__range / 2.

    # should an error be thrown if low < 0?

    if inp_isarray:
        return np.random.uniform(low, high)
    else:
        return np.random.uniform(low, high, array_len)


def seed(seed=1):
    """
    Set the C++, the python and the numpy random seed to desired value

    :param seed: Random number generator should be seeded by this value.
        Default is 1
    """

    cy_helpers.srand(seed)
    random.seed(seed)
    np.random.seed(seed)
