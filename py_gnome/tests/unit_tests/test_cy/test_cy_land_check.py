# -*- coding: utf-8 -*-

"""
Tests of the cython land-check code used in the map code.

Designed to be run with py.test

@author: Chris.Barker
"""


import pytest

import numpy as np
from gnome.cy_gnome.cy_land_check import overlap_grid, find_first_pixel


class Test_overlap_grid(object):

    m = 100
    n = 200

    def test_right(self):
        """line totally to the right of the grid"""

        pt1 = (101, 13)
        pt2 = (101, 220)
        assert not overlap_grid(self.m, self.n, pt1, pt2)

    def test_left(self):
        pt1 = (-1, 13)
        pt2 = (-1, 220)
        assert not overlap_grid(self.m, self.n, pt1, pt2)

    def test_over(self):
        pt1 = (10, 210)
        pt2 = (100, 220)
        assert not overlap_grid(self.m, self.n, pt1, pt2)

    def test_under(self):
        pt1 = (10, -1)
        pt2 = (100, -23)
        assert not overlap_grid(self.m, self.n, pt1, pt2)

    def test_inside(self):
        pt1 = (10, 50)
        pt2 = (90, 150)
        assert overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top(self):
        pt1 = (50, 50)
        pt2 = (199, 201)
        assert overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top_right_corner(self):
        pt1 = (95, 205)
        pt2 = (105, 195)
        assert overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top_right_corner2(self):
        pt2 = (95, 205)
        pt1 = (105, 195)
        assert overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_lower_left(self):
        pt2 = (-1, 3)
        pt1 = (3, -1)
        assert overlap_grid(self.m, self.n, pt1, pt2)

    def test_from_lower_left(self):
        pt2 = (-1, -1)
        pt1 = (2, 3)
        assert overlap_grid(self.m, self.n, pt1, pt2)


hit_examples = [((5, 5), (15, 5), (9, 5), (10, 5)), ((15, 5), (5, 5),
                (11, 5), (10, 5)), ((0, 0), (10, 5), (9, 4), (10, 5)),
                ((19, 0), (0, 9), (11, 4), (10, 4))]


@pytest.mark.parametrize(('pt1', 'pt2', 'res1', 'res2'), hit_examples)
def test_land_cross(
    pt1,
    pt2,
    res1,
    res2,
    ):
    """
    try a single LE that should be crossing land
    """

    # a very simple raster:

    (w, h) = (20, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny vertical line:

    raster[10, :] = 1

    # pt1 = ( 5, 5)
    # pt2 = (15, 5)

    result = find_first_pixel(raster, pt1, pt2)

    print(result)
    assert result[0] == res1
    assert result[1] == res2

no_hit_examples = [((5, 5), (9, 5)), ((15, 5), (11, 5)), ((0, 0), (9,
                   9))]

@pytest.mark.parametrize(('pt1', 'pt2'), no_hit_examples)
def test_land_not_cross(pt1, pt2):
    """
    try a single LE that should be crossing land
    """

    # a very simple raster:

    (w, h) = (20, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny vertical line:

    raster[10, :] = 1

    result = find_first_pixel(raster, pt1, pt2)

    assert result is None

diag_hit_examples = [((3, 0), (0, 3), (2, 1), (1, 1)), ((0, 3), (3, 0),
                     (1, 2), (2, 2)), ((4, 1), (0, 4), (3, 2), (2, 2)),
                     ((0, 4), (4, 1), (2, 3), (3, 3)), ((0, 8), (9, 0),
                     (3, 5), (4, 4))]

@pytest.mark.parametrize(('pt1', 'pt2', 'prev_pt', 'hit_pt'),
                         diag_hit_examples)
def test_land_cross_diag(
    pt1,
    pt2,
    prev_pt,
    hit_pt,
    ):
    """
    try a single LE that should be crossing land
    with a diagonal land line..
    """

    # a very simple raster:

    (w, h) = (10, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny digonal line part way:

    raster[0, 0] = 1
    raster[1, 1] = 1
    raster[2, 2] = 1
    raster[3, 3] = 1
    raster[4, 4] = 1

    result = find_first_pixel(raster, pt1, pt2)

    print(pt1, pt2, prev_pt, hit_pt)
    print(result)
    assert result[0] == prev_pt
    assert result[1] == hit_pt

diag_not_hit_examples = [((9, 1), (1, 9)), ((1, 9), (9, 1))]


@pytest.mark.parametrize(('pt1', 'pt2'), diag_not_hit_examples)
def test_land_not_cross_diag(pt1, pt2):
    """
    try a single LE that should be crossing land
    with a diagonal land line..
    """

    # a very simple raster:

    (w, h) = (10, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny diagonal line part way:

    raster[0, 0] = 1
    raster[1, 1] = 1
    raster[2, 2] = 1
    raster[3, 3] = 1
    raster[4, 4] = 1

    result = find_first_pixel(raster, pt1, pt2)

    print(pt1, pt2, result)
    assert result is None

points_outside_grid = [((-5, -5), (25, 15), (9, 4), (10, 5))]  # outside from right


@pytest.mark.parametrize(('pt1', 'pt2', 'prev_pt', 'hit_pt'),
                         points_outside_grid)
def test_points_outside_grid(
    pt1,
    pt2,
    prev_pt,
    hit_pt,
    ):
    """
    try a single LE that should be crossing land
    with points starting or ending outside grid
    """

    # a very simple raster:

    (w, h) = (20, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny vertical line:

    raster[10, :] = 1

    result = find_first_pixel(raster, pt1, pt2)

    print(pt1, pt2, prev_pt, hit_pt)
    print(result)
    assert result[0] == prev_pt
    assert result[1] == hit_pt

## starting on the raster, heading way off
## This was an bug/issue found when we had a bad velocity value
## it should never happened, but shouldn't fail, either...
way_off_examples = [(1000, 1000), # not that big!
                    (2147483647, 2147483647), # as big as it gets (32 bit)
                    (-2147483647, 2147483647), # as big as it gets (32 bit)
                    (2147483647, -2147483647), # as big as it gets (32 bit)
                    (-2147483647, -2147483647), # as big as it gets (32 bit)
                   ]
@pytest.mark.parametrize('pt2', way_off_examples)
def test_way_off(pt2):
    # a very simple raster -- no land
    (w, h) = (20, 20)
    raster = np.zeros((w, h), dtype=np.uint8)
    # a little block in the middle
    # raster[8:12, 8:12] = 1
    pt1 = (10, 10)
    print(pt1)
    print(pt2)
    result = find_first_pixel(raster, pt1, pt2)

    assert result is None

off_to_off_examples = [( (-10, -10), (30, 30) ),
                       ( (30, 30), (-10, -10) ),
                       ( (30, -10), (-10, 30) ),
                       ( (-2, 40), (30, -10) ),
                       ]

@pytest.mark.parametrize(('pt1','pt2'), off_to_off_examples)
def test_off_to_off(pt1, pt2):
    """test  movement that is entirely off the raster"""
    # a very simple raster -- no land
    (w, h) = (20, 20)
    raster = np.zeros((w, h), dtype=np.uint8)

    result = find_first_pixel(raster, pt1, pt2)

    assert result is None

# def test_outside_raster(self):
#         """
#         test LEs starting from outside the raster bounds
#         """
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ((-50, -30), (-50, 30),
#                                       (50, 30), (50, -30)),
#                         projection=projections.NoProjection(),
#                         )
#
#         # one left to right
#         # one right to left
#         # diagonal that doesn't hit
#         # diagonal that does hit
#         spill = gnome.spills.Spill(num_LEs=4)
#         spill['positions']= np.array(((30.0, 5.0, 0.0), # outside from right
#                                      (-5.0, 5.0, 0.0), # outside from left
#                                      (5.0, -5.0, 0.0), # outside from top
#                                      (-5.0, -5.0, 0.0), # outside from upper left
#                                      ), dtype=np.float64)
#
#         spill['next_positions'] =  np.array( ( (  15.0, 5.0, 0.0),
#                                      (  5.0, 5.0, 0.0),
#                                      (  5.0, 15.0, 0.0 ),
#                                      ( 25.0, 15.0, 0.0 ),
#                                      ),  dtype=np.float64)
#
#         map.beach_elements(spill)
#
#         assert np.array_equal( spill['next_positions'], ( ( 15.0, 5.0, 0.0),
#                                                      ( 5.0, 5.0, 0.0),
#                                                      ( 5.0, 15.0, 0.0),
#                                                      (10.0, 5.0, 0.0),
#                                                      ) )
#         # just the beached ones
#         assert np.array_equal(spill['last_water_positions'][3:],
#                               ((9.0, 4.0, 0.0),))
#
#         assert np.array_equal(spill['status_codes'][3:],
#                               (basic_types.oil_status.on_land,))

if __name__ == '__main__':


    test_way_off( (-30000000, -30000000) )

    # (w, h) = (10, 10)
    # raster = np.zeros((w, h), dtype=np.uint8)

    # # a single skinny diagonal line part way:

    # raster[0, 0] = 1
    # raster[1, 1] = 1
    # raster[2, 2] = 1
    # raster[3, 3] = 1
    # raster[4, 4] = 1

    # pt1 = (0, 8)
    # pt2 = (9, 0)

    # print 'checking:'
    # print pt1
    # print pt2
    # result = find_first_pixel(raster, pt1, pt2)

    # print result

    # pt1 = (pt1[1], pt1[0])
    # pt2 = (pt2[1], pt2[0])

    # print 'checking:'
    # print pt1
    # print pt2
    # result = find_first_pixel(raster, pt1, pt2)

    # print result
