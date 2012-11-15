# -*- coding: utf-8 -*-
"""

Tests of the cython land-check code used in the map code.

Designed to be run with py.test


@author: Chris.Barker

"""

import numpy as np
from gnome.cy_gnome import cy_land_check as land_check
# first test the python version:
# from gnome import land_check

class Test_overlap_grid():
    m = 100
    n = 200

    def test_right(self):
        """line totally to the right of the grid"""
        pt1 = (101, 13)
        pt2 = (101, 220)
        assert not land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_left(self):
        pt1 = (-1, 13)
        pt2 = (-1, 220)
        assert not land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_over(self):
        pt1 = (10,  210)
        pt2 = (100, 220)
        assert not land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_under(self):
        pt1 = (10,  -1)
        pt2 = (100, -23)
        assert not land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_inside(self):
        pt1 = (10, 50)
        pt2 = (90, 150)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top(self):
        pt1 = (50, 50)
        pt2 = (199, 201)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top_right_corner(self):
        pt1 = (95, 205)
        pt2 = (105, 195)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_top_right_corner2(self):
        pt2 = (95, 205)
        pt1 = (105, 195)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)

    def test_cross_lower_left(self):
        pt2 = (-1, 3)
        pt1 = (3, -1)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)
    def test_from_lower_left(self):
        pt2 = (-1, -1)
        pt1 = (2, 3)
        assert land_check.overlap_grid(self.m, self.n, pt1, pt2)


# from gnome import basic_types        
# class Test_full_move:
#     """
#     A test to see if the full API is working for beaching
    
    
#     It should check for land-jumping and return the "last known water point" 
#     """
#     # a very simple raster:
#     w, h = 20, 10
#     raster = np.zeros((w, h), dtype=np.uint8)
#     # a single skinny vertical line:
#     raster[10, :] = 1
        
#     def test_on_map(self):
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
#         # making sure the map is set up right
#         assert map.on_map((100.0, 1.0)) is False
#         assert map.on_map((0.0, 1.0))

#     def test_on_land(self):
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
#         assert map.on_land( (10, 3, 0) ) == 1 
#         assert map.on_land( (9,  3, 0) ) == 0
#         assert map.on_land( (11, 3, 0) ) == 0
        

#     def test_land_cross(self):
#         """
#         try a single LE that should be crossing land
#         """
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
         
#         spill = gnome.spill.Spill(num_LEs=1)

#         spill['positions'] = np.array( ( ( 5.0, 5.0, 0.0), ), dtype=np.float64) 
#         spill['next_positions'] = np.array( ( (15.0, 5.0, 0.0), ), dtype=np.float64)
#         spill['status_codes'] = np.array( (basic_types.oil_status.in_water, ), dtype=basic_types.status_code_type)
 
#         map.beach_elements(spill)
        
#         assert np.array_equal( spill['next_positions'][0], (10.0, 5.0, 0.0) )
#         assert np.array_equal( spill['last_water_positions'][0], (9.0, 5.0, 0.0) )
#         assert spill['status_codes'][0] == basic_types.oil_status.on_land
        
#     def test_land_cross_array(self):
#         """
#         test a few LEs
#         """
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
        
#         # one left to right
#         # one right to left
#         # one diagonal upper left to lower right
#         # one diagonal upper right to lower left
        
#         spill = gnome.spill.Spill(num_LEs=4)

#         spill['positions'] = np.array( ( ( 5.0, 5.0, 0.0),
#                                          ( 15.0, 5.0, 0.0),
#                                          ( 0.0, 0.0, 0.0),
#                                          (19.0, 0.0, 0.0),
#                                          ), dtype=np.float64) 
#         spill['next_positions'] = np.array( ( ( 15.0, 5.0, 0.0),
#                                               (  5.0, 5.0, 0.0),
#                                               ( 10.0, 5.0, 0.0),
#                                               (  0.0, 9.0, 0.0 ),
#                                               ),  dtype=np.float64) 
#         map.beach_elements(spill)

#         assert np.array_equal( spill['next_positions'], ( (10.0,  5.0, 0.0),
#                                                      (10.0, 5.0, 0.0),
#                                                      (10.0, 5.0, 0.0),
#                                                      (10.0, 4.0, 0.0),
#                                                      ) )
        
#         assert np.array_equal( spill['last_water_positions'], ( (9.0,  5.0, 0.0),
#                                                                 (11.0, 5.0, 0.0),
#                                                                 ( 9.0, 4.0, 0.0),
#                                                                 (11.0, 4.0, 0.0),
#                                                                 ) )

#         assert np.alltrue( spill['status_codes']  ==  basic_types.oil_status.on_land )

#     def test_some_cross_array(self):
#         """
#         test a few LEs
#         """
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
        
#         # one left to right
#         # one right to left
#         # diagonal that doesn't hit
#         # diagonal that does hit
 
#         spill = gnome.spill.Spill(num_LEs=4)

#         spill['positions'] = np.array( ( ( 5.0, 5.0, 0.0),
#                                          (15.0, 5.0, 0.0),
#                                          ( 0.0, 0.0, 0.0),
#                                          (19.0, 0.0, 0.0),
#                                          ), dtype=np.float64) 

#         spill['next_positions'] = np.array( ( (  9.0, 5.0, 0.0),
#                                               ( 11.0, 5.0, 0.0),
#                                               (  9.0, 9.0, 0.0),
#                                               (  0.0, 9.0, 0.0),
#                                               ),  dtype=np.float64)
 
#         map.beach_elements(spill)

#         assert np.array_equal( spill['next_positions'], ( ( 9.0, 5.0, 0.0),
#                                                      (11.0, 5.0, 0.0),
#                                                      ( 9.0, 9.0, 0.0),
#                                                      (10.0, 4.0, 0.0),
#                                                      ) )
#         # just the beached ones
#         assert np.array_equal( spill['last_water_positions'][3:], ( (11.0, 4.0, 0.0),
#                                                                     ) )

#         assert np.array_equal( spill['status_codes'][3:], ( basic_types.oil_status.on_land,
#                                                             ) )   


#     def test_outside_raster(self):
#         """
#         test LEs starting form outside the raster bounds
#         """
#         map = RasterMap(refloat_halflife = 6, #hours
#                         bitmap_array= self.raster,
#                         map_bounds = ( (-50, -30), (-50, 30), (50, 30), (50, -30) ),
#                         projection=projections.NoProjection(),
#                         )
        
#         # one left to right
#         # one right to left
#         # diagonal that doesn't hit
#         # diagonal that does hit
#         spill = gnome.spill.Spill(num_LEs=4)
#         spill['positions']= np.array( ( ( 30.0, 5.0, 0.0), # outside from right
#                                      ( -5.0, 5.0, 0.0), # outside from left
#                                      ( 5.0, -5.0, 0.0), # outside from top
#                                      ( -5.0, -5.0, 0.0), # outside from upper left
#                                      ), dtype=np.float64) 

#         spill['next_positions'] =  np.array( ( (  15.0, 5.0, 0.0),
#                                      (  5.0, 5.0, 0.0),
#                                      (  5.0, 15.0, 0.0 ),
#                                      ( 25.0, 15.0, 0.0 ),
#                                      ),  dtype=np.float64)
        
#         map.beach_elements(spill)
        
#         assert np.array_equal( spill['next_positions'], ( ( 15.0, 5.0, 0.0),
#                                                      ( 5.0, 5.0, 0.0),
#                                                      ( 5.0, 15.0, 0.0),
#                                                      (10.0, 5.0, 0.0),
#                                                      ) )
#         # just the beached ones
#         assert np.array_equal( spill['last_water_positions'][3:], ( ( 9.0, 4.0, 0.0),
#                                                                     ) )

#         assert np.array_equal( spill['status_codes'][3:], ( basic_types.oil_status.on_land,
#                                                             ) )




if __name__ == "__main__":
     tester = Test_full_move()
     tester.test_land_cross()



        
