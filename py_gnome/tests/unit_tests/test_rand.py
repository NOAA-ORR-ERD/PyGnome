#!/usr/bin/env python
"""
Test gnome.utilities.rand.py
"""

import numpy as np
import random

from gnome.utilities.rand import random_with_persistance, seed
from gnome.cy_gnome.cy_helpers import rand

import pytest


def test_exceptions():
    """
    Test exceptions
    """
    with pytest.raises(ValueError):
        random_with_persistance(1, 0)
        random_with_persistance(1, 0, 0, 0)


def test_random_with_persistance_scalar():
    """
    Since numbers are randomly generated, can only test
    array length
    """
    x = random_with_persistance(0, 10)
    assert x >= 0 and x <= 10
    assert isinstance(x, float)


def test_random_with_persistance_array():
    """
    Test so the output is a numpy array of random numbers
    """
    x = random_with_persistance(0, 10, 900, 60, 100)
    assert len(x) == 100
    assert isinstance(x, np.ndarray)


def test_random_with_persistance_low_equals_high():
    """
    if low==high, then return low - deterministic output
    """
    x = random_with_persistance(10, 10, 900, 60)
    assert x == 10

    y = random_with_persistance(10, 10, 900, 60, array_len=100)
    assert len(y) == 100
    assert np.all(y == 10)


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
