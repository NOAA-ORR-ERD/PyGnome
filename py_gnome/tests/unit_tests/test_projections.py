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

## tests for meters_to_latlon
m2l = projections.FlatEarthProjection.meters_to_latlon

def test_meters_to_latlon():
    """ distance at equator """        
    assert np.allclose( m2l( (111195.11, 111195.11), 0.0 ),
                           (1.0, 1.0) )
     
def test_meters_to_latlon2():
    """ distance at 60 deg north (1/2) """
    assert np.allclose( m2l( (111195.11, 111195.11), 60.0 ),
                           (2.0, 1.0) )
     
def test_meters_to_latlon3():
    """ distance at 90 deg north: it should get very large!"""

    dlonlat = m2l( (111195.11, 111195.11), 90.0 )
    assert dlonlat[0,0] > 1e16  # somewhat arbitrary...it should be infinity, but apparently not with fp rounding

## test the geodesic on the sphere code:     
geodesic_sphere = projections.FlatEarthProjection.geodesic_sphere
def test_near_equatorE():
    """ directly east on the equator """
    lon, lat = geodesic_sphere(30, 0.0, 111195.11, 90.0)
    print lon, lat
    assert round(lon, 6) == 31.0
    assert round(lat, 15) == 0.0
     
def test_near_equatorW():
    """ directly west on the equator """
    lon, lat = geodesic_sphere(-30.0, 0.0, 111195.11, 270.0)
    print lon, lat
    assert round(lon, 6) == -31.0
    assert round(lat, 15) == 0.0
     
def test_near_equatorN():
    """ directly north on the equator """
    lon, lat = geodesic_sphere(30, 0.0, 111195.11, 0.0)
    print lon, lat
    assert round(lon, 6) == 30.0
    assert round(lat, 6) == 1.0
     
def test_near_equatorS():
    """ directly south on the equator """
    lon, lat = geodesic_sphere(30, 0.0, 111195.11, 180.0)
    print lon, lat
    assert round(lon, 6) == 30.0
    assert round(lat, 6) == -1.0
     
def test_near_equatorNE():
    """ directly northeast from the equator """
    lon, lat = geodesic_sphere(0.0, 0.0, 111195.11, 45.0)
    print lon, lat
    # these values from the online geodesic calculator (which uses and elipsoidal earth)
    #   http://geographiclib.sourceforge.net/cgi-bin/Geod
    acc = 2 # almost good to 3 decimal place -- still not great!
    assert round(lon, acc) == round(0.70635273, acc)
    assert round(lat, acc) == round(0.71105843, acc) 
     
def test_north_NE():
    """ directly northeast from north of the equator """
    lon, lat = geodesic_sphere(0.0, 60.0, 111195.11, 45.0)
    print lon, lat
    # these values from the online geodesic calculator (which uses and elipsoidal earth)
    #   http://geographiclib.sourceforge.net/cgi-bin/Geod
    acc = 2 # almost good to 3 decimal place -- still not great!
    assert round(lon, acc) == round(1.43959 , acc)
    assert round(lat, acc) == round(60.69799, acc) 
     
     
     