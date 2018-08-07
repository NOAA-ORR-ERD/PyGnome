"""
geometry package

This package has:

A few higher-level objects for geometry: a Bounding Box class and a Polygon class.

It also has some lover level code basic geometry that acts on numpy arrays of points:

i.e. a polygon is expressed as a Nx2 numpy array of float64

Some of these are in Cython for speed.
"""

from .cy_point_in_polygon import point_in_poly, points_in_poly

from .poly_clockwise import is_clockwise_convex, is_clockwise

