#!/usr/bin/env python


"""
tests of the gnome.utilities.projections module
"""
import numpy as np
from gnome.utilities import projections

def test_NoProjection():
    proj = projections.NoProjection()
    
    coords = ( (45.5, 32.1),
               (-18.3, -12.6),
               )
    result = [[ 45,  32],
              [-18, -12]]
    proj_coords = proj.to_pixel(coords)
    print proj_coords
    
    assert np.array_equal(proj_coords, result)

    proj_coords = proj.to_pixel(coords, asint=False)
    print proj_coords

    assert np.array_equal(proj_coords, coords)


class Test_GeoProjection():
    bounding_box = ( (-10.0, 23.0),
                     (-5,  33.0) )
    
    image_size = (500, 500)

    proj = projections.GeoProjection(bounding_box, image_size)
    
    def test_bounds(self):

        # corners of the BB:
        coords  = ( (-10.0, 23.0),
                     (-5,  33.0),
                     (-10, 33),
                     (-5, 23.0) )
        
        proj_coords = self.proj.to_pixel(coords, asint=True)
        
        print proj_coords
    
        assert np.array_equal(proj_coords, [[  125, 500],
                                            [ 375,   0],
                                            [ 125,   0],
                                            [ 375, 500]])
        
    def test_middle(self):
        #middle of the BB
        coords  = ( (-7.5, 28.0),
                    )
        
        proj_coords = self.proj.to_pixel(coords, asint=True)
        
        print proj_coords
    
        assert np.array_equal(proj_coords, [[  250, 250]] )
        
    def test_outside(self):
        """
        points outside the BB should still come back, but the pixel coords
        will be outside the image shape
        """ 
        #just outside the bitmap
        coords  = ( (-12.500001, 22.99999999),
                    ( -2.499999, 33.00000001),
                    )
        
        print "scale:",  self.proj.scale

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print proj_coords
    
        assert np.array_equal(proj_coords, [[ -1, 500],
                                            [500,  -1]]
                              )
    
    def test_reverse(self):
        
        coords  = ( (-7.5, 28.0), #middle of the BB
                    (-8.123, 25.345), # some non-round numbers
                    ( -2.500001, 33.00000001), # outside the bitmap
                    ( -4.2345,   32.123),
                    )
        # first test without rounding
        proj_coords = self.proj.to_pixel(coords, asint=False)
        back_coords = self.proj.to_lat_long(proj_coords)
        print coords
        print proj_coords
        print repr(back_coords)
        assert np.allclose(coords, back_coords)
        
        # now with the pixel rounding
        proj_coords = self.proj.to_pixel(coords, asint=True)
        back_coords = self.proj.to_lat_long(proj_coords)
        print coords
        print proj_coords
        print repr(back_coords)
        
        # tolerence set according to the scale:
        tol = 1. / self.proj.scale[0] / 1.5 # should be about 1/2 pixel
        assert np.allclose(coords, back_coords, rtol = 1e-100, atol=tol) # rtol tiny so it really doesn't matter
    
    
class Test_FlatEarthProjection():
    # bb with 60 degrees in the center: ( cos(60 deg) == 0.5 )
    bounding_box = ( ( 20, 50.0 ),
                     ( 40, 70.0 ) )
    
    image_size = (500, 500)

    proj = projections.FlatEarthProjection(bounding_box, image_size)
    
    def test_bounds(self):

        # corners of the BB: (sqare in lat-long)
        coords  = ( (20.0, 50.0),
                    (20.0, 70.0),
                    (40.0, 50.0),
                    (40.0, 70.0) )
        
        proj_coords = self.proj.to_pixel(coords, asint=True)
        
        print proj_coords
    
        assert np.array_equal(proj_coords, [[ 124, 500],
                                            [ 124,   0],
                                            [ 375, 500],
                                            [ 375,   0]])
        
    def test_middle(self):
        #middle of the BB
        coords  = ( (30, 60.0),
                    )
        
        proj_coords = self.proj.to_pixel(coords, asint=True)
        
        print proj_coords
    
        assert np.array_equal(proj_coords, [[  250, 250]] )
        
    def test_outside(self):
        """
        points outside the BB should still come back, but the pixel coords
        will be outside the image shape
        """ 
        #just outside the bitmap
        coords  = ( (  9.9999999, 49.99999999),
                    ( 50.0000001, 70.000000001),
                    )
        
        print "scale:",  self.proj.scale

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print proj_coords
    
        assert np.array_equal(proj_coords, [[ -1, 500],
                                            [500,  -1]]
                              )
    
    def test_reverse(self):
        
        coords  = ( (-7.5, 28.0), #middle of the BB
                    (-8.123, 25.345), # some non-round numbers
                    ( -2.500001, 33.00000001), # outside the bitmap
                    ( -4.2345,   32.123),
                    )
        # first test without rounding
        proj_coords = self.proj.to_pixel(coords, asint=False)
        back_coords = self.proj.to_lat_long(proj_coords)
        print coords
        print proj_coords
        print repr(back_coords)
        assert np.allclose(coords, back_coords)
        
        # now with the pixel rounding
        proj_coords = self.proj.to_pixel(coords, asint=True)
        back_coords = self.proj.to_lat_long(proj_coords)
        print coords
        print proj_coords
        print repr(back_coords)
        
        # tolerence set according to the scale:
        tol = 1. / self.proj.scale[0] / 1.5 # should be about 1/2 pixel
        assert np.allclose(coords, back_coords, rtol = 1e-100, atol=tol) # rtol tiny so it really doesn't matter
    
     