#!/usr/bin/env python

"""
test_PinP

Test code for Point in Polygon module

Designed to be run with py.test
"""

import numpy as np

from gnome.utilities.geometry import PinP

# the very simplest test
poly1 = np.array(((0, 0),
                  (0, 1),
                  (1, 1),
                  (1, 0),
                  ), np.float)

# this one has first and last points duplicated
poly2 = np.array(((0, 0),
                  (0, 1),
                  (1, 1),
                  (1, 0),
                  (0, 0),
                  ), np.float)


def test_inside1():
    assert PinP.CrossingsTest(poly1, (0.5, 0.5)) is True


def test_inside2():
    assert PinP.CrossingsTest(poly2, (0.5, 0.5)) is True


def test_on_vertex():
    assert PinP.CrossingsTest(poly1, (1, 1)) is True


def test_outside1():
    assert PinP.CrossingsTest(poly1, (2, 2)) is False


def test_outside2():
    assert PinP.CrossingsTest(poly2, (2, 2)) is False


def test_float():
    poly = ((-50, -30), (-50, 30), (50, 30), (50, -30))
    poly = np.array(poly, dtype=np.float64)

    #assert PinP.CrossingsTest( poly, (0, 0) ) is True
    assert PinP.CrossingsTest(poly, (100.0, 1.0)) is False

## test the points in polygon code:


def test_points_in_poly_scalar():

    assert PinP.points_in_poly(poly1, (0.5, 0.5, 0.0)) is True

    assert PinP.points_in_poly(poly1, (1.5, 0.5, 0.0)) is False


def test_points_in_poly_array_one_element():

    assert np.array_equal(PinP.points_in_poly(poly1, ((0.5, 0.5, 0.0), )),
                          np.array((True,))
                          )

    assert np.array_equal(PinP.points_in_poly(poly1, ((1.5, -0.5, 0.0), )),
                          np.array((False,))
                          )


def test_points_in_poly_array():
    points = np.array(((0.5, 0.5, 0.0),
                       (1.5, 0.5, 0.0),
                       (0.5, 0.5, 0.0),
                       (0.5, 0.5, 0.0),
                       (-0.5, 0.5, 0.0),
                       (0.5, 0.5, 0.0),))

    result = np.array((True,
                       False,
                       True,
                       True,
                       False,
                       True,
                       ))

    assert np.array_equal(PinP.points_in_poly(poly2, points),
                          result)
