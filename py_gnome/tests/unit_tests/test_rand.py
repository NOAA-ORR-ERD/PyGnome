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
                         [(1, 0),
                          (1., 0.),
                          ([1, 2], [0, 1])])
def test_exceptions(low, high):
    """
    Test exceptions
    """

    with pytest.raises(ValueError):
        random_with_persistance(low, high)


@pytest.mark.parametrize(("low", "high", "persist", "array_len"),
                   [(0, 10, None, None),
                    (0, 10,  0, 10),
                    ([1, 3], [5, 6], None, None),
                    ((1, 3), (5, 6), (0, 0), 5),  # should ignore array_len
                    (np.asarray([1., 3.]), np.asarray([4., 7.]), None, None)])
def test_random_with_0_persistance(low, high, persist, array_len):
    """
    Since numbers are randomly generated, can only test
    array length
    """
    x = random_with_persistance(low, high, persist, array_len=array_len)
    if isinstance(low, int) or isinstance(low, float):
        if (array_len is None):
            assert x >= low and x <= high
            assert isinstance(x, float)
            return
        else:
            assert len(x) == array_len

    # assertion if an array is returned
    assert np.all(x >= low) and np.all(x <= high)


@pytest.mark.parametrize(("low", "high", "persist"),
                   [(1, 4, 100),
                    ((1, 3), (5, 6), (0, 100)),  # should ignore array_len
                    ])
def test_random_with_non_zero_persistence(low, high, persist):
    """
    Checks that everything works with non-zero persistence
    For an array, it also checks that if an element has non-zero persistence,
    the random number generated for it is within its [low, high] bounds
    """
    x = random_with_persistance(low, high, persist)
    if isinstance(persist, tuple):
        mask = (persist == 0)
        assert x[mask] >= low[mask]
        assert x[mask] <= high[mask]

    assert True


@pytest.mark.parametrize(("low", "high"), [(10, 10), ([0, 1], [0, 1])])
def test_random_with_persistance_low_equals_high(low, high):
    """
    if low==high, then return low - deterministic output
    """
    x = random_with_persistance(low, high, 900, 60)
    if isinstance(low, int) or isinstance(low, float):
        assert x is low
    else:
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
