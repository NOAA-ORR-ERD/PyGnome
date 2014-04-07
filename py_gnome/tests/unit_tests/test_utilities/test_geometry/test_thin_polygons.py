#!/usr/bin/env python

"""
Tests of the polygon thinning code

"""

import numpy as np

from gnome.utilities.geometry.polygons import Polygon, PolygonSet

from gnome.utilities.file_tools.haz_files import ReadBNA
# load up a set of polygons

polys439 = ReadBNA('00439polys_013685pts.bna', polytype = "PolygonSet")

poly1 = PolygonSet()
points1 = ((0,     0),
          (0,    95),# these two very close together
          (5,   100),
          (100, 100),# these two a bit further apart, but still very close
          (105,  10),
          (100,   0))
poly1.append( points1 )

points2 = ((150,  80),
           (150, 100),
           (160, 100),
           (160, 80),
           (150, 80), # first and last the same!
           )

poly2 = PolygonSet()
poly2.append(points2)


def test_no_thin1():
    """ scaling to 1 shouldn't thin the polygon """
    thinned = poly1.thin(scale=(1, 1) )

    assert thinned == poly1

def test_thin11():
    """ scaled enough to thin out one point """
    thinned = poly1.thin(scale=(.1, .1) )

    assert len(thinned[0]) == 5
    assert np.array_equal( thinned[0][-3:], points1[-3:] )
def test_thin12():
    """ scaled enough to thin out two points """
    thinned = poly1.thin(scale=(.05, .05) )

    assert len(thinned[0]) == 4
    assert np.array_equal( thinned[0][:2], points1[:2] )
    assert np.array_equal( thinned[0][-2:], points1[3:5] )

def test_no_thin2():
    """ slight scaling shouldn't change it """
    thinned = poly2.thin(scale=(0.9, 0.8) )

    assert thinned == poly2

def test_thin21():
    """ scaled enough to thin out one point """
    thinned = poly2.thin(scale=(.05, .05) )

    assert len(thinned[0]) == 3
    assert np.array_equal( thinned[0][:2], points2[:2] )
    assert np.array_equal( thinned[0][0], thinned[0][-1] ) # first and last should be same.

#    assert False

def test_thin23():
    """ scaled enough to thin out to zero points """
    thinned = poly2.thin(scale=(.01, .01) )

    assert len(thinned) == 0






