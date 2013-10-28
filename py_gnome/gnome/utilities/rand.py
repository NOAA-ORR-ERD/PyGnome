"""
rand.py

Contains functions for adding randomness - not to
confuse with standard python random functions
"""

import numpy as np
from math import sqrt
from gnome.cy_gnome import cy_helpers
import random


def random_with_persistance(
    low,
    high,
    array=None,  # update this array, if provided
    persistence=None,
    time_step=1.,
    ):
    """
    Used by gnome to generate a randomness between low and high, which is
    persistent for duration time_step

    :param low: lower bound for random number; should be an array, tuple, list
    :param high: upper bound for random number; should be an array, tuple, list
    :param array: array to be updated. Must be same length as 'low', 'high',
        'persistence'. Default is None in which case the computed array is
        simply returned
    :param time_step: step size for the simulation in seconds.
    :param persistence: in seconds. Since we add randomness for each timestep,
        the persistence parameter is used to make the randomness invariant to
        size of time_step. Default is None. If persistence is None, it gets set
        equal to 'time_step'. If persistence < 0 for any elements, their values
        are not updated in the 'array'

    :returns: returns 'array' with newly computed values

    Note: persistence and time_step should be in the same time units
          Assumes both low and high are int/float or arrays, lists, tuples.
          For lists, arrays, tuples - it converts input to numpy array with
          dtype=float

          If 'low' is an array, list or tuple, then 'high' and 'persistence'
          should also be the same length array, list or tuple. Give
          all 3 parameters for each element of the array.
    """

    # make copies since we don't want to change the original arrays
    low = np.copy(low)
    high = np.copy(high)

    if array is None:
        array = np.zeros(len(low,), dtype=float)
    else:
        if not isinstance(array, np.ndarray):
            raise ValueError("If an 'array' is provided for computed values,"
                    " it must be a numpy array")

    # exceptions
    len_msg = ("Length of 'low', 'high' and 'persistence' arrays"
               " should be equal")

    if persistence is not None:
        persistence = np.asarray(persistence)
        if (len(high) != len(persistence)):
            raise ValueError(len_msg)

    if (len(high) != len(low)):
        raise ValueError(len_msg)

    if np.any(high < low):
        raise ValueError('The lower bound for random_with_persistance must be'
                '  less than or equal to upper bound for all array elements')

    if np.all(low == high):
        array[:] = low[:]

    if persistence is None:
        """
        if persistence == time_step, then no need to scale the [low, high]
        interval
        """
        array[:] = np.random.uniform(low, high)
        print array
    else:
        """
        if persistence == time_step, then no need to scale the [low, high]
        interval
        """
        u_mask = (persistence > 0)  # update mask for values to be changed

        if np.any(u_mask):
            if np.any(persistence[u_mask] != time_step):
                """
                only need to do the following for persistence values !=
                time_step. For persistence == time_step, the newly computed
                'low' and 'high' are unchanged so it is alright to recompute.
                Recomputing for elements with persistence == time_step for
                numpy arrays should still be very efficient and code is more
                readable.
                """
                orig = high[u_mask] - low[u_mask]
                l__range = orig * np.sqrt(persistence[u_mask] / float(time_step))
                mean = (high[u_mask] + low[u_mask]) / 2.

                # update the bounds for generating the random number
                low[u_mask] = mean - l__range / 2.
                high[u_mask] = mean + l__range / 2.

            array[u_mask] = np.random.uniform(low[u_mask], high[u_mask])

    return array


def seed(seed=1):
    """
    Set the C++, the python and the numpy random seed to desired value

    :param seed: Random number generator should be seeded by this value.
        Default is 1
    """

    cy_helpers.srand(seed)
    random.seed(seed)
    np.random.seed(seed)
