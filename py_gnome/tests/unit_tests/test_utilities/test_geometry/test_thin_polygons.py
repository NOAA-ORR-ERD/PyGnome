#!/usr/bin/env python

"""
Tests of the polygon thinning code

"""
import os

import numpy as np

from gnome.utilities.geometry.polygons import Polygon, PolygonSet

from gnome.utilities.file_tools.haz_files import ReadBNA



## some test polygons:
points1 = ((0,     0),
          (0,    95),# these two very close together
          (5,   100),
          (100, 100),# these two a bit further apart, but still very close
          (105,  10),
          (100,   0))
poly1 = Polygon(points1, metadata={'name':'poly1'})

points2 = ((150,  80),
           (150, 100),
           (160, 100),
           (160, 80),
           (150, 80), # first and last the same!
           )

poly2 = Polygon(points2, metadata={'name':'poly2'})

# this one  needs to get scaled up...
points3 = ((0.0, 0.0),
           (0.1, 0.3),
           (0.2, 0.3),
           (0,    0),
           )

poly3 = Polygon(points3, metadata={'name':'poly3'})


def test_no_thin1():
    """ scaling to 1 shouldn't thin the polygon """
    thinned = poly1.thin(scale=(1, 1) )

    assert thinned == poly1

def test_thin11():
    """ scaled enough to thin out one point """
    thinned = poly1.thin(scale=(.1, .1) )

    assert len(thinned) == 5
    assert np.array_equal( thinned[-3:], points1[-3:] )
def test_thin12():
    """ scaled enough to thin out two points """
    thinned = poly1.thin(scale=(.05, .05) )

    assert len(thinned) == 4
    assert np.array_equal( thinned[:2], points1[:2] )
    assert np.array_equal( thinned[-2:], points1[3:5] )

def test_no_thin2():
    """ slight scaling shouldn't change it """
    thinned = poly2.thin(scale=(0.9, 0.8) )

    assert thinned == poly2

def test_thin21():
    """ scaled enough to thin out one point """
    thinned = poly2.thin(scale=(.05, .05) )

    assert len(thinned) == 4
    assert np.array_equal( thinned[:2], points2[:2] )
    assert np.array_equal( thinned[0], thinned[-1] ) # first and last should be same.

def test_thin23():
    """ scaled enough to thin out to zero points """
    thinned = poly2.thin(scale=(.01, .01) )

    assert len(thinned) == 0

def test_no_thin3():
    """ just enough scaling shouldn't change it """
    thinned = poly3.thin(scale=(10, 10) )

    assert thinned == poly3

def test_thin32():
    """ some scaling should remove one point """
    thinned = poly3.thin(scale=(5.1, 5.1) )

    assert len(thinned) == 3
    # start and end points should still be same
    assert np.array_equal( thinned[0], thinned[-1] )

## test on PolygonSet:

pset = PolygonSet()
pset.append(poly1)
pset.append(poly2)
pset.append(poly3)

def test_thin_set():
    thinned = pset.thin( scale=(0.1, 0.1) )

    assert len(thinned) == 2
    assert thinned[0] == poly1.thin( scale=(0.1, 0.1) )
    assert thinned[1] == poly2.thin( scale=(0.1, 0.1) )

def test_larger():
    filename = os.path.join( os.path.split(__file__)[0],'00439polys_013685pts.bna' )

    polys439 = ReadBNA(filename, polytype = "PolygonSet")

    thinned = polys439.thin( scale=(264, 300) ) # somewhat scaled for latitude

    ## note: I don't know if this is right -- but it is somewhat smaller, but not totally thinned
    assert len(thinned) == 413



