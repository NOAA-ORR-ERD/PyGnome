#!/usr/bin/env python

"""
Cython code to call C point in poly routine

Should I just port the C to Cython???
"""


import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as cnp

# declare the interface to the C code
cdef extern char c_point_in_poly1(size_t nvert, double *vertices, double *point)


@cython.boundscheck(False)
@cython.wraparound(False)
def point_in_poly(cnp.ndarray[double, ndim=2, mode="c"] poly not None,
                   in_point):
    """
    point_in_poly( poly, in_point )

    Determines if point is in the polygon -- 1 if it is, 0 if not

    :param poly: A Nx2 numpy array of doubles.
    :param point: A (x,y) sequence of floats (doubles, whatever)

    NOTE: points on the boundary are arbitrarily (fp errors..),
          but consistently considered either in or out, so that a given point
          should be in only one of two polygons that share a boundary.

    This calls C code I adapted from here:
    http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
    """
    cdef size_t nvert
    cdef char result
    cdef double[2] point

    point[0] = in_point[0]
    point[1] = in_point[1]

    nvert = poly.shape[0]

    result = c_point_in_poly1(nvert, &poly[0, 0], point)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def points_in_poly(cnp.ndarray[double, ndim=2, mode="c"] pgon, points):
    """
    compute whether the points given are in the polygon defined in pgon.

    :param pgon: the vertices of teh polygon
    :type pgon: NX2 numpy array of floats

    :param points: the points to test
    :type points: NX3 numpy array of (x, y, z) floats

    :returns: a boolean array the same length as points
              if the input is a single point, the result is a
              scalr python boolean

    Note: this version takes a 3-d point, even though the third coord
          is ignored.
    """

    np_points = np.ascontiguousarray(points, dtype=np.float64)
    scalar = (np_points.shape == (3,))
    np_points.shape = (-1, 3)

    cdef double [:, :] a_points
    a_points = np_points

    ## fixme -- proper way to get np.bool?
    cdef cnp.ndarray[char, ndim = 1, mode = "c"] result = np.zeros((a_points.shape[0],), dtype=np.uint8)

    cdef unsigned int i, nvert, npoints

    nvert = pgon.shape[0]
    npoints = a_points.shape[0]

    for i in range(npoints):
        result[i] = c_point_in_poly1(nvert, &pgon[0, 0], &a_points[i, 0])
    if scalar:
        return bool(result[0])  # to make it a regular python bool
    else:
        return result.view(dtype=np.bool)  # make it a np.bool array
