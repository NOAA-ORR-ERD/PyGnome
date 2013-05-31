#!/usr/bin/env python

"""
Some tests of the geometry functions

Designed to be run with py.test

"""
import numpy as np

#from py_ugrid.geometry.point_in_polygon import point_in_poly
from py_ugrid.geometry.cy_point_in_polygon import point_in_poly

import pytest

class Test_point_in_polygon():
    """
    test various known polygons and points, both clockwise and anti-clockwise

    NOTE: need to do tests that assure that all points are in only one polygon
          i.e. points on a line will be one or the other, but not both.
    """
    ## counter clockwise polygon:
    polygon_ccw = np.array(( ( -5, -2),
                             (  3, -1),
                             (  5, -1),
                             (  5,  4),
                             (  3,  0),
                             (  0,  0),
                             ( -2,  2),
                             ( -5,  2),
                             ), dtype = np.float64 )

    ## clockwise polygon:
    polygon_cw = polygon_ccw[::-1].copy()
    
    points_in_poly  = [( (-3.0, 0.0 ), ),
                       ( ( 4.0, 0.0 ), ),
                       #( ( -3.0,  2.0 ),  ), # on a horizontal line top
                       #( (  5.0,  2.0 ),  ), # on a vertical line on right
                       #( ( -1.0, -1.0 ),  ), # diagonal line on right 
                       ]
    @pytest.mark.parametrize(("point",), points_in_poly )
    def test_point_in_poly(self, point):
        """
        tests points that should be in the polygon
        """
        assert point_in_poly(self.polygon_ccw, point)
        #assert point_in_poly2(self.polygon_ccw, point)
    @pytest.mark.parametrize(("point",), points_in_poly )
    def test_point_in_poly2(self, point):
        """
        tests points that should be in the polygon -- clockwise polygon
        """
        assert point_in_poly(self.polygon_cw, point)


    points_outside_poly  = [( ( 0.0, -5.0 ), ), # below
                            ( ( 9.0, 0.0 ), ),  # right
                            ( (-9.0, 0.0 ), ),  # left
                            ( ( 0.0, 5.0 ), ),  # above
                            ( ( 0.0, -3.0), ),  # directly below a vertex
                            ( ( -6.0, 2.0 ), ),  # along horizontal line outside
                            ( (  0.0, 2.0 ), ),  # along horizontal line outside
                            ( (  6.0, 2.0 ), ),  # along horizontal line outside
                            ( (  1.0, 2.0 ), ), # in the "bay"
                            ( (  3.0, 3.0 ), ), # in the "bay"
                            #( (  4.0,  2.0 ),  ), # diagonal line on left 
                            #( (  -5.0,  0.0 ),  ), # on a vertical line on left
                            #( (  4.0, -1.0 ),  ), # on a horizontal line bottom
                            ]
    
    @pytest.mark.parametrize(("point",), points_outside_poly )
    def test_point_outside_poly(self, point):
        """
        tests points that should be outside the polygon
        """
        assert not point_in_poly(self.polygon_ccw, point)
        #assert not point_in_poly2(self.polygon_ccw, point)

    @pytest.mark.parametrize(("point",), points_outside_poly )
    def test_point_outside_poly2(self, point):
        """
        tests points that should be outside the polygon -- clockwise polygon
        """
        assert not point_in_poly(self.polygon_cw, point)
    
    # vertices = [ ( polygon_ccw[0],  ),
    #              ( polygon_ccw[1],  ), 
    #              ( polygon_ccw[2],  ), 
    #              ( polygon_ccw[3],  ), 
    #              ( polygon_ccw[4],  ), 
    #              ( polygon_ccw[5],  ), 
    #              ( polygon_ccw[6],  ), 
    #              ]

    # @pytest.mark.parametrize(("vertex",), vertices )
    # def test_vertices():
    #     """
    #     What should the vertex behaviour be???
    #     """



# class Test_point_in_triangle():
#     """
#     test the point_in_tri code
#     """
#     ## CCW
#     triangle_ccw = np.array(( (-2, -2),
#                           ( 3, 3),
#                           ( 0, 4) ),
#                         dtype = np.float64 )

#     ## CW
#     triangle_cw = triangle_ccw[::-1]
    
#     points_in_tri  = [( ( 0.0, 1.0 ), ),
#                       ( (-1.0, 0.0 ), ),
#                       ( triangle_cw[0],  ), # the vertices
#                       ( triangle_cw[1],  ), # the vertices
#                       ( triangle_cw[2],  ), # the vertices
#                       ( ( 0.0, 0.0), ), # on the line
#                       ( (-1.0, 1.0), ), # on the line
#                       ( ( 1.5, 3.5), ), # on the line
#                       ]
#     @pytest.mark.parametrize(("point",), points_in_tri )
#     def test_point_in_tri(self, point):
#         """
#         tests points that should be in the triangle
#         """
#         assert point_in_tri(self.triangle_cw, point)
#         assert point_in_tri(self.triangle_ccw, point)

#     points_not_in_tri  = [ ( (5.0, 0.0 ), ),
#                            ( (-5.0, 0.0 ), ),
#                            ( ( 0.0000000001, -0.0000000001), ), # just outside the line
#                            ( (-1.0000000001, 1.00000000001), ), # just outside the line
#                            ( ( 1.5000000001, 3.5), ), # just outside the line
#                            ( ( 5.0,  5.0), ), # outside, but aligned with a side
#                            ( ( 6.0,  2.0), ), # outside, but aligned with a side
#                            ( (-3.0, -5.0), ), # outside, but aligned with a side
#                             ]
#     @pytest.mark.parametrize(("point",), points_not_in_tri )
#     def test_point_not_in_tri(self, point):
#         """
#         tests points that should be in the grid
#         """
#         assert not point_in_tri(self.triangle_cw, point)
#         assert not point_in_tri(self.triangle_ccw, point)



