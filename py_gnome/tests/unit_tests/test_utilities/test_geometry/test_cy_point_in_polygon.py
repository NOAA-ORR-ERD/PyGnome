#!/usr/bin/env python

"""
Some tests of the point in polygon functions in the cython code.

Designed to be run with py.test

"""

import pytest

import numpy as np

## the Cython version:

from gnome.utilities.geometry.cy_point_in_polygon import point_in_poly, \
    points_in_poly

poly1_ccw = np.array((
    (-5, -2),
    (3, -1),
    (5, -1),
    (5, 4),
    (3, 0),
    (0, 0),
    (-2, 2),
    (-5, 2),
    ), dtype=np.float64)

poly1_cw = poly1_ccw[::-1].copy()

# shares part of the boundary with poly1_ccw

poly2_ccw = np.array((
    poly1_ccw[3],
    poly1_ccw[4],
    poly1_ccw[5],
    poly1_ccw[6],
    poly1_ccw[7],
    (-3, 5),
    (1, 4),
    (4, 6),
    ))

poly2_cw = poly2_ccw[::-1].copy()

points_in_poly1 = [((-3.0, 0.0), ), ((4.0, 0.0), ), ((4.5, 2.5), )]

                  #  ( polygon_ccw[0],  ), # the vertices
                  #  ( polygon_ccw[1],  ), # the vertices
                  #  ( polygon_ccw[2],  ), # the vertices
                  #  ( polygon_ccw[3],  ), # the vertices
                  #  ( polygon_ccw[4],  ), # the vertices
                  #  ( polygon_ccw[5],  ), # the vertices
                  #  ( polygon_ccw[6],  ), # the vertices
                  #  ( ( -3.0,  2.0 ),  ), # on a horizontal line top
                  #  ( (  5.0,  2.0 ),  ), # on a vertical line on right
                  #  ( ( -1.0, -1.0 ),  ), # diagonal line on right

points_in_poly2 = [((2.0, 3.0), ), ((-3.0, 2.1), ), ((4.0, 5.0), )]

points_on_boundaries = [
    ((4.0, 2.0), ),
    ((3.25, 0.5), ),
    ((1.5, 0.0), ),
    ((-1.0, 1.0), ),
    ((-0.5, 0.5), ),
    ((-0.1, 0.1), ),
    ((-3.0, 2.0), ),
    ]

points_on_vertices = [(poly1_ccw[4], ), (poly1_ccw[5], ),
                      (poly1_ccw[6], )]  # the shared vertices


@pytest.mark.parametrize(('point', ), points_in_poly1)
def test_point_in_poly(point):
    """
    tests points that should be in the polygon
    """

    assert point_in_poly(poly1_ccw, point)
    assert point_in_poly(poly1_cw, point)


@pytest.mark.parametrize(('point', ), points_in_poly1)
def test_point_not_in_poly2(point):
    """
    points that are in poly1 should not be in poly2
    """

    assert not point_in_poly(poly2_ccw, np.asarray(point,
                             dtype=np.float64))
    assert not point_in_poly(poly2_cw, np.asarray(point,
                             dtype=np.float64))


@pytest.mark.parametrize(('point', ), points_in_poly2)
def test_point_in_poly3(point):
    """
    tests points that should be in the polygon
    """

    assert point_in_poly(poly2_ccw, np.asarray(point, dtype=np.float64))
    assert point_in_poly(poly2_cw, np.asarray(point, dtype=np.float64))


@pytest.mark.parametrize(('point', ), points_in_poly2)
def test_point_not_in_poly1(point):
    """
    points that are in poly1 should not be in poly2
    """

    assert not point_in_poly(poly1_ccw, np.asarray(point,
                             dtype=np.float64))
    assert not point_in_poly(poly1_cw, np.asarray(point,
                             dtype=np.float64))


@pytest.mark.parametrize(('point', ), points_on_boundaries)
def test_point_on_boundary(point):
    """
    tests points that are on the boundaries between two polygons
    it should be computed as being only in one and only one of them
    """

    p1 = point_in_poly(poly1_ccw, np.asarray(point, dtype=np.float64))
    p2 = point_in_poly(poly2_ccw, np.asarray(point, dtype=np.float64))
    assert p1 ^ p2  # bitwise xor -- these should be integer 1 or 0


@pytest.mark.parametrize(('point', ), points_on_vertices)
def test_point_on_vertex(point):
    """
    tests points that are on the vertex between two polygons
    it should be computed as being only in one and only one of them
    """

    p1 = point_in_poly(poly1_ccw, np.asarray(point, dtype=np.float64))
    p2 = point_in_poly(poly2_ccw, np.asarray(point, dtype=np.float64))
    print
    assert p1 ^ p2  # bitwise xor -- these should be integer 1 or 0


def test_non_contiguous_poly():
    """
    a non-contiguous poly should fail
    """

    poly = np.zeros((5, 4), dtype=np.float64)

    # make non-contiguous

    poly = poly[:, 2:]
    print poly.flags

    with pytest.raises(ValueError):
        point_in_poly(poly, (3, 4))


#    @pytest.mark.parametrize(("point",), points_in_poly )
#    def test_point_in_poly2(self, point):
#        """
#        tests points that should be in the polygon -- clockwise polygon
#        """
#        assert point_in_poly(self.polygon_cw,
#                             np.asarray(point, dtype=np.float64))
#
#
#    points_outside_poly = [((0.0, -5.0), ),  # below
#                           ((9.0, 0.0), ),  # right
#                           ((-9.0, 0.0), ),  # left
#                           ((0.0, 5.0), ),  # above
#                           ((0.0, -3.0), ),  # directly below a vertex
#                           ((-6.0, 2.0), ),  # along horizontal line outside
#                           ((0.0, 2.0), ),  # along horizontal line outside
#                           ((6.0, 2.0), ),  # along horizontal line outside
#                           ((1.0, 2.0), ),  # in the "bay"
#                           ((3.0, 3.0), ),  # in the "bay"
#                           ((4.0, 2.0), ),  # diagonal line on left
#                           ((-5.0, 0.0), ),  # on a vertical line on left
#                           ((4.0, -1.0), ),  # on a horizontal line bottom
#                           ]
#
#
#    @pytest.mark.parametrize(("point",), points_outside_poly)
#    def test_point_outside_poly(self, point):
#        """
#        tests points that should be outside the polygon
#        """
#        assert not point_in_poly(self.polygon_ccw, point)
#        #assert not point_in_poly2(self.polygon_ccw, point)
#
#
#    @pytest.mark.parametrize(("point",), points_outside_poly)
#    def test_point_outside_poly2(self, point):
#        """
#        tests points that should be outside the polygon -- clockwise polygon
#        """
#        assert not point_in_poly(self.polygon_cw, point)
#
#
# class Test_point_in_triangle():
#    """
#    test the point_in_tri code
#    """
#    ## CCW
#    triangle_ccw = np.array(((-2, -2),
#                             (3, 3),
#                             (0, 4) ),
#                             dtype=np.float64)
#
#    ## CW
#    triangle_cw = triangle_ccw[::-1]
#
#    points_in_tri = [((0.0, 1.0 ), ),
#                     ((-1.0, 0.0 ), ),
#                     (triangle_cw[0], ),  # the vertices
#                     (triangle_cw[1], ),  # the vertices
#                     (triangle_cw[2], ),  # the vertices
#                     (( 0.0, 0.0), ),  # on the line
#                     ((-1.0, 1.0), ),  # on the line
#                     (( 1.5, 3.5), ),  # on the line
#                     ]
#
#
#    @pytest.mark.parametrize(("point",), points_in_tri)
#    def test_point_in_tri(self, point):
#        """
#        tests points that should be in the triangle
#        """
#        assert point_in_tri(self.triangle_cw, point)
#        assert point_in_tri(self.triangle_ccw, point)
#
#    points_not_in_tri = [((5.0, 0.0 ), ),
#                         ((-5.0, 0.0 ), ),
#                         ((0.0000000001, -0.0000000001), ),  # just outside the line
#                         ((-1.0000000001, 1.00000000001), ),  # just outside the line
#                         ((1.5000000001, 3.5), ),  # just outside the line
#                         ((5.0, 5.0), ),  # outside, but aligned with a side
#                         ((6.0, 2.0), ),  # outside, but aligned with a side
#                         ((-3.0, -5.0), ),  # outside, but aligned with a side
#                         ]
#
#
#    @pytest.mark.parametrize(("point",), points_not_in_tri)
#    def test_point_not_in_tri(self, point):
#        """
#        tests points that should be in the grid
#        """
#        assert not point_in_tri(self.triangle_cw, point)
#        assert not point_in_tri(self.triangle_ccw, point)
#

## test the points in polygon code:

poly1 = np.array(((0, 0), (0, 1), (1, 1), (1, 0)), np.float64)


def test_points_in_poly_scalar():
    assert points_in_poly(poly1, (0.5, 0.5, 0.0)) is True
    assert points_in_poly(poly1, (1.5, 0.5, 0.0)) is False


def test_points_in_poly_array_one_element():
    assert np.array_equal(points_in_poly(poly1, ((0.5, 0.5, 0.0), )),
                          np.array((True, )))
    assert np.array_equal(points_in_poly(poly1, ((1.5, -0.5, 0.0), )),
                          np.array((False, )))


def test_points_in_poly_array():
    points = np.array((
        (0.5, 0.5, 0.0),
        (1.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        ))

    result = np.array((
        True,
        False,
        True,
        True,
        False,
        True,
        ))

    assert np.array_equal(points_in_poly(poly1, points), result)


