# -*- coding: utf-8 -*-
"""
Latest Revision: Mon Mar 21 11:00:00 PST 2012

@author: brian.zelenke
You should  be able to `git checkout prototype; cd py_gnome; python setup.py build; python setup2.py develop` and then run this script.
"""

from __future__ import division
import numpy as np
import gnome.map


def test_map_island_color():
    '''
    Test the creation of a color map with an island inset.
    '''
    m = gnome.map.gnome_map([500,500],"MapBounds_Island.bna",'RGB') #Create a 500x500 pixel map.
    m.image.save('Color_MapBounds.png') #Write the result to the present working directory as a PNG image file.
    assert True
    #assert False #Force this test to fail so that NOSE will print output to the command window.

def test_map_island_monochrome():
    '''
    Test the creation of a black and white map with an island inset.
    '''
    m = gnome.map.lw_map([500,500],"MapBounds_Island.bna",2.*60.*60.,"1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    m.image.save('B&W_MapBounds.png') #Write the result to the present working directory as a PNG image file.
    assert True
    #assert False #Force this test to fail so that NOSE will print output to the command window.

def test_map_in_water():
    '''
    Test whether the location of a particle on the map -- in or out of water -- is determined correctly.
    '''
    m = gnome.map.lw_map([500,500],"MapBounds_Island.bna",2.*60.*60.,"1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    
    #Coordinate of a point within the water area of MapBounds_Island.bna.
    LatInWater=48.1647
    LonInWater=-126.78709
    
    assert(m.in_water((LonInWater,LatInWater))) #Throw an error if the know in-water location returns false.

def test_map_on_land():
    '''
    Test whether the location of a particle on the map -- off or on land -- is determined correctly.
    '''
    m = gnome.map.lw_map((500,500), "MapBounds_Island.bna", 2.*60.*60., "1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    
    #Coordinate of a point on the island of MapBounds_Island.bna.
    OnLand = (-126.78709, 47.833333)
    
    assert(m.on_land( OnLand )) #Throw an error if the know on-land location returns false.

def test_in_water_resolution():
    '''
    Test the limits of the precision, to within an order of magnitude, defining whether a point is in or out of water.
    '''
    
    m = gnome.map.lw_map([500,500],"MapBounds_Island.bna",2.*60.*60.,"1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    
    #Specify coordinates of the two points that make up the southeastern coastline segment of the island in the BNA map.
    x1=-126.78709
    y1=47.666667
    x2=-126.44218
    y2=47.833333
    
    #Define a point on the line formed by this coastline segment.
    slope=(y2-y1)/(x2-x1)
    b=y1-(slope*x1)
    py=47.7
    px=(py-b)/slope
    
    #Find the order of magnitude epsilon change in the latitude that causes the
    #given point to "move" from water to land.
    eps=np.spacing(1) #Distance between 1 and the nearest floating point number.
    mag=0.
    running=True
    while running:
        mag=mag+1.0
        print "Order of magnitude: %g" %mag
        running=m.in_water((px,py+(eps*(10.0**mag))))
    
    #Difference in position within an order of magnitude in degrees of latitude necessary to "move" point from water to land.
    dlatO0=(eps*(10.0**(mag-1.0)))
    dlatO1=(eps*(10.0**mag))
    
    print "A particle positioned on a coastline segment must be moved something more than %g meters, but less than %g meters, inland before pyGNOME acknowledges it's no longer in water." %(dlatO0*1852.0,dlatO1*1852.0)
