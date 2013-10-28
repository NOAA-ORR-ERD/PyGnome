#!/usr/bin/env python

"""
Test gnome.utilities.rand.py
"""

import numpy as np
import random

from gnome.utilities.rand import random_with_persistance, seed
from gnome.cy_gnome.cy_helpers import rand

import pytest


@pytest.mark.parametrize(("low", "high"),
                         [([1], [0]),
                          ([1, 2], [0, 1])])
def test_exceptions(low, high):
    """
    Test exceptions
    """
    arr = np.zeros((len(low),), dtype=np.float64)
    with pytest.raises(ValueError):
        random_with_persistance(low, high, arr)


@pytest.mark.parametrize(("low", "high", "array", "persist", "timestep"),
                   [([0], [10], None, None, 1),
                    ([1, 3], [5, 6], None, None, 1),
                    ((1, 3), (5, 6), np.zeros((2,), dtype=float), (-1, 1), 1),
                    (np.asarray([1., 3.]), np.asarray([4., 7.]),
                     np.zeros((2,), dtype=np.float64),
                     np.asarray([100, 0]), 100)
                     ])
def test_random_with_persistance(low, high, array, persist, timestep):
    """
    Since numbers are randomly generated, can only test
    array length
    """
    if array is None:
        array = random_with_persistance(low, high, array, persist,
                                        time_step=timestep)
    else:
        random_with_persistance(low, high, array, persist, time_step=timestep)

    # persist set to timestep
    if persist is None:
        assert np.all(array >= low) and np.all(array <= high)
    else:
        # do not update values if persist < 0
        low = np.asarray(low)
        high = np.asarray(high)
        persist = np.asarray(persist)
        assert array[persist <= 0] == 0

        # for persist == timestep, output is within [low, high] bounds
        eq_mask = [persist == timestep]
        if np.any(eq_mask):
            assert (array[eq_mask] >= low[eq_mask] and
                    array[eq_mask] <= high[eq_mask])


@pytest.mark.parametrize(("low", "high"),
                         [([10], [10]),
                          ([0, 1], [0, 1])])
def test_random_with_persistance_low_equals_high(low, high):
    """
    if low==high, then return low - deterministic output
    """
    x = random_with_persistance(low, high)
    assert np.all(x == low)


def test_set_seed():
    """
    test set_seed to 1 works
    """

    seed(1)

    xi = [random.uniform(0, i + 1) for i in range(10)]
    ai = np.random.uniform(0, 1, 10)
    ci = [rand() for i in range(10)]

    seed(1)
    xf = [random.uniform(0, i + 1) for i in range(10)]
    af = np.random.uniform(0, 1, 10)
    cf = [rand() for i in range(10)]

    assert xi == xf
    assert np.all(ai == af)
    assert ci == cf
