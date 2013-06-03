#!/usr/bin/env python

"""
Some test of the py_ugrid classes

Designed to be run with py.test

"""
import numpy as np

import pytest

from py_ugrid import ugrid, grid_examples

class Test_locate_simple_grid():
    """
    initilizing about a simple a grid as you can with data
    """
    nodes = [(0,0),
             (2,0),
             (1,2),
             (3,2)]
    faces = [(0, 1, 2),
             (1, 3, 2),]
    edges = [(0,1),
             (1,3),
             (3,2),
             (2,0)]
    
    grid = ugrid.ugrid(nodes, faces, edges)
    
    points_in_grid  = [( (1.0,  1.0 ), 0),
                       ( (1.45, 1.48), 1),
                       ( (1.9999999,  0.00000001), 0), # almost on a point
                       ( (2.0000000,  0.00000001), 1), # almost on a point
                       ( (1.4999999999, 1.0), 0), # almost on an edge
                       ]
    
    @pytest.mark.parametrize(("point", "face_num"), points_in_grid )
    def test_point_in_grid(self, point, face_num):
        """
        tests points that should be in the grid
        """
        print "point is:", point
        assert self.grid.locate_face_simple(point) == face_num


    points_off_grid  = [( (0.0,  1.0 ), ),
                        ( (2.65, 1.0), ),
                        ( (1.5, 2.00001), ),
                        ( (1.0, -0.0000001), ),
                       ]
    
    @pytest.mark.parametrize(("point", ), points_off_grid )
    def test_point_off_grid(self, point):
        """
        tests point that should be off the grid
        """
        assert self.grid.locate_face_simple(point) is None
    
class Test_locate_simple_grid2():
    """
    initilizing about a simple a grid as you can with data
    """

    grid = grid_examples.get_triangle_2()
    
    points_in_grid  = [( (7.07,  2.08 ), 0),
                       ( (9.5,  6.0 ), 12),
                       ( (7.55,  13.5 ), 20),
                       ]
    
    @pytest.mark.parametrize(("point", "face_num"), points_in_grid )
    def test_point_in_grid(self, point, face_num):
        """
        tests points that should be in the grid
        """
        assert self.grid.locate_face_simple(point) == face_num


    points_off_grid  = [( (3.5,  1.7 ), ),
                        ( (11.5,  2.5 ), ),
                        ( (7.5,  5.5 ), ), # in the hole
                       ]
    
    @pytest.mark.parametrize(("point", ), points_off_grid )
    def test_point_off_grid(self, point):
        """
        tests point that should be off the grid
        """
        assert self.grid.locate_face_simple(point) is None
    
    ## these points are on nodes or baoundaries -- more than one answer is valid.
    ##   different algorithms may yield different results   
    points_on_boundaries  = [ ( (5.0, 7.0    ), (6, 7, 8, 9) ), # exactly on a point
                              ( (5.0, 3.1    ), (1, 2) ), # exactly on a vertical line
                              ( (10.5, 4.0    ), (4, 11) ), # exactly on a horizonal line
                              ( (6.0, 8.0    ), (9, 10) ), # exactly on a diagonal line
                              ( (4.0, 4.0    ), (1, 5) ), # exactly on a diagonal line
                              ( (11.0, 7.0    ), (11, 12, 14) ), # exactly on a boundary node (should None be in this list?)
                              ( (3.0, 7.0    ), (5, 6, 7, None) ), # exactly on a boundary node
                              ( (9.0, 7.0    ), (12, 13, 14, 15, None) ), # exactly on a boundary node
                              ( (5.0, 5.0    ), (1, 2, 5, 6, 8, None) ), # exactly on a boundary node
                              ]
    
#    @pytest.mark.parametrize(("point", "face_nums"), points_on_boundaries )
#    def test_point_in_grid(self, point, face_nums):
#        """
#        tests points that should be in the grid
#        """
#        assert self.grid.locate_face_simple(point) in face_nums
