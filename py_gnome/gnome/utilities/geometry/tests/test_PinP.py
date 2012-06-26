#!/usr/bin/env python

"""
test_PinP

Test code for Point in Polygon module

Designed to be run with py.test
"""

import numpy as np

## test in_place
import sys
sys.path.insert(0, "../")

import PinP

# the very simplest test
poly1 = np.array( ( (0,0),
                    (0,1),
                    (1,1),
                    (1,0),
                    ), np.float )

# this one has first and last points duplicated
poly2 = np.array( ( (0,0),
                    (0,1),
                    (1,1),
                    (1,0),
                    (0,0),
                    ), np.float )


def test_inside1():
    assert PinP.CrossingsTest( poly1, (0.5, 0.5) ) is True

def test_inside2():
    assert PinP.CrossingsTest( poly2, (0.5, 0.5) ) is True

def test_on_vertex():
    assert PinP.CrossingsTest( poly1, (1, 1) ) is True

def test_outside1():
    assert PinP.CrossingsTest( poly1, (2, 2) ) is False

def test_outside2():
    assert PinP.CrossingsTest( poly2, (2, 2) ) is False

def test_float():
    poly = ( (-50, -30), (-50, 30), (50, 30), (50, -30) )
    poly = np.array(poly, dtype = np.float64)
    
    #assert PinP.CrossingsTest( poly, (0, 0) ) is True
    
    assert PinP.CrossingsTest( poly, (100.0, 1.0) ) is False

               