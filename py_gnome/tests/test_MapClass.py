# -*- coding: utf-8 -*-
"""
Latest Revision: Mon Mar 05 15:00:00 PST 2012

@author: brian.zelenke
You should  be able to `git checkout prototype; cd py_gnome; python setup.py build; python setup2.py develop` and then run this script.
"""

import gnome.map



def test_map_island_color():
    '''
    Test the creation of a color map with an island inset.
    '''
    m = gnome.map.gnome_map([500,500],"MapBounds_Island.bna",'RGB') #Create a 500x500 pixel map.
    m.image.save('Color_MapBounds.png') #Write the result to the present working directory as a PNG image file.
    assert False #Force this test to fail so that NOSE will print output to the command window.

def test_map_island_monochrome():
    '''
    Test the creation of a black and white map with an island inset.
    '''
    m = gnome.map.lw_map([500,500],"MapBounds_Island.bna",2.*60.*60.,"1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    m.image.save('B&W_MapBounds.png') #Write the result to the present working directory as a PNG image file.
    assert False #Force this test to fail so that NOSE will print output to the command window.

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
    m = gnome.map.lw_map([500,500],"MapBounds_Island.bna",2.*60.*60.,"1") #Create a 500x500 pixel map, with an LE refloat half-life of 2 hours (specified here in seconds).
    
    #Coordinate of a point on the island of MapBounds_Island.bna.
    LatOnLand=47.833333
    LonOnLand=-126.78709
    
    assert(m.on_land((LonOnLand,LatOnLand))) #Throw an error if the know on-land location returns false.
