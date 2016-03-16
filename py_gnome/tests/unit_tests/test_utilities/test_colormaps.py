"""
Tests of the gd map canvas code.
"""
import numpy as np

import pytest

from gnome.utilities.colormaps import ColorMap, NamedColorMaps


def test_init():
    """
    Can we even initialize one?
    """
    cm = ColorMap("hsv")
    assert np.all(cm.get_colors((0, 100, 255)) == [[255, 0, 0],
                                                   [0, 255, 81],
                                                   [255, 0, 24]])


def test_init_with_val_range():
    """
    Can we initialize with a val_range?
    """
    cm = ColorMap("hsv", val_range=(0, 511))
    assert np.all(cm.get_colors((0, 200, 511)) == [[255, 0, 0],
                                                   [0, 255, 81],
                                                   [255, 0, 24]])


def test_integer_val_range():
    """
        Can we set an integer val_range, and does that range accurately map
        to the color positions?
    """
    cm = ColorMap("hsv")

    # testing our entire range.
    # our range is 1000 + (256 - 1) * 4 which should match the size of the
    # colormap
    cm.val_range = (1000, 2020)
    for i, v in enumerate(range(1000, 2024, 4)):
        assert np.all(NamedColorMaps['hsv'][i] == cm.get_colors((v,)))


def test_float_val_range():
    """
        Can we set a floating point val_range, and does that range accurately
        map to the color positions?
    """
    cm = ColorMap("hsv")

    # testing our entire range.
    cm.val_range = (10.0, 20.0)

    for i, v in enumerate(np.linspace(10.0, 20.0, 256)):
        assert np.all(NamedColorMaps['hsv'][i] == cm.get_colors((v,)))


def test_reversed_float_val_range():
    """
        Can we set a reversed floating point val_range, and does that range
        accurately map to the color positions?
    """
    cm = ColorMap("hsv")

    # testing our entire range.
    cm.val_range = (20.0, 10.0)

    for i, v in enumerate(np.linspace(20.0, 10.0, 256)):
        assert np.all(NamedColorMaps['hsv'][i] == cm.get_colors((v,)))
