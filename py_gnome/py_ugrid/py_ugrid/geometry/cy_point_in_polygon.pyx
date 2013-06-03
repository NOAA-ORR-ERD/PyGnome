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
cdef extern int c_point_in_poly1(size_t nvert, double *vertices, double *point)

@cython.boundscheck(False)
@cython.wraparound(False)
def point_in_poly( cnp.ndarray[double, ndim=2, mode="c" ] poly not None,
                   in_point ):
    """
    point_in_poly( poly, point )
    
    Determines if point is in the polygon -- 1 if it is, 0 if not
    
    :param poly: A Nx2 numpy array of doubles.
    :param point: A (x,y) sequence of floats (doubles, whatever)    
    
    NOTE: points on the boundary are arbitrarily (fp errors..), but consistently
          considered either in or out, so that a given point should
          be in only one of two polygons that share a boundary.
    
    This calls C code I adapted from here:
    http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html    
    """
    cdef size_t nvert
    cdef int result
    
    cdef cnp.ndarray[double, ndim=1, mode="c" ] point
    point = np.asarray(in_point, dtype=np.float64).reshape((2,))

    nvert = poly.shape[0]
    
    result = c_point_in_poly1(nvert, &poly[0,0], &point[0])
    
    return result

