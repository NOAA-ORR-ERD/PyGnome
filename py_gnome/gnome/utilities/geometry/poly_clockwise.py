#!/usr/bin/env python

"""
code for checking if a polygon is cockwise or counter-clockwise

There are two versions:

is_clockwise_convex only works for convex polygons -- but is faster,
it only needs to check three points.

is_clockwise checks all points, but works for convex and cocave
  (note: that's the same as the area calculation)

from:
http://paulbourke.net/geometry/clockwise/

"""


def is_clockwise(poly):
    """
    returns True if the polygon is clockwise ordered, false if not

    expects a sequence of tuples, or something like it (Nx2 array for instance),
    of the points:

    [ (x1, y1), (x2, y2), (x3, y3), ...(xi, yi) ]

    See: http://paulbourke.net/geometry/clockwise/
    """

    total = poly[-1][0] * poly[0][1] - poly[0][0] * poly[-1][1]  # last point to first point
    for i in xrange(len(poly) - 1):
        total += poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]

    if total <= 0:
        return True
    else:
        return False


def is_clockwise_convex(poly):
    """
    returns True if the polygon is clockwise ordered, false if not

    expects a sequence of tuples, or something like it, of the points:

    [ (x1, y1), (x2, y2), (x3, y3), ...(xi, yi) ]

    This only works for concave polygons. See:

    http://paulbourke.net/geometry/clockwise/
    """

    x0 = poly[0][0]
    y0 = poly[0][1]
    x1 = poly[1][0]
    y1 = poly[1][1]
    x2 = poly[2][0]
    y2 = poly[2][1]

    cp = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
    if cp <= 0:
        return True
    else:
        return False
