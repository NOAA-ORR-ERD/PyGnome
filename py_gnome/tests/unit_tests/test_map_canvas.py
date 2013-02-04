"""
Tests of the map canvas code.
"""

import pytest

import os
import numpy as np
import numpy.random as random

from gnome import basic_types
from gnome.spill_container import TestSpillContainer
from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
import gnome.utilities.geometry.polygons

datadir = os.path.join(os.path.dirname(__file__), r"SampleData")

def test_render():
    """
    tests the ability to render a basemap from a bna
    """
    map_canvas.make_map(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "map_sample.png")

    assert True

def test_render_BW():
    '''
    Test the creation of a black and white map with an island inset.

    note: it is rendered "almost black" on black...
    '''
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    m = map_canvas.BW_MapCanvas( (500,500) )
    m.set_land(polygons)
    m.draw_background()
    #m.save_background('BW_LandMap.png') #Write the result to the present working directory as a PNG image file.
    
    assert True


def test_basemap_square():
    map = map_canvas.MapCanvas((300, 300))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map.set_land(set)
    map.draw_background()
    map.save_background("background_square.png")
    assert True

def test_basemap_square2():
    map = map_canvas.MapCanvas((100, 300))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map.set_land(set)
    map.draw_background()
    map.save_background("background_square2.png")
    assert True

def test_basemap_square3():
    map = map_canvas.MapCanvas((300, 100))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map.set_land(set)
    map.draw_background()
    map.save_background("background_square3.png")
    assert True


def test_basemap():
    map = map_canvas.MapCanvas((400, 800))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    map.set_land(polygons)
    map.draw_background()
    map.save_background("background1.png")
    assert True

def test_basemap_wide():
    map = map_canvas.MapCanvas((800, 400))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    map.set_land(polygons)
    map.draw_background()
    map.save_background("background2.png")
    
    assert True

def test_render_elements():
    map = map_canvas.MapCanvas((800, 600))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    map.set_land(polygons)
    
    BB = map.map_BB
    min_lon, min_lat = BB[0] 
    max_lon, max_lat = BB[1] 
    
    N = 1000
    #create some random particle positions:
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    #create a spill
    spill = TestSpillContainer(num_elements=N)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat

    map.create_foreground_image()
    map.draw_elements(spill)

    # create an uncertainty spill
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    spill = TestSpillContainer(num_elements=N, uncertain=True)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat

    map.draw_elements(spill)

    # save the image
    map.save_foreground("foreground1.png")
    assert True        


