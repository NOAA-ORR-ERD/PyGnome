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

basedir = os.path.dirname(__file__)
datadir = os.path.join(basedir, r"../SampleData")

def test_render():
    """
    tests the ability to render a basemap from a bna
    """
    map_canvas.make_map(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), 
                        os.path.join(basedir,"map_sample.png"))

    assert True

def test_render_BW():
    '''
    Test the creation of a black and white map with an island inset.

    note: it is rendered "almost black" on black...
    '''
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    m = map_canvas.BW_MapCanvas( (500,500), land_polygons=polygons)
    m.draw_background()
    #m.save_background('BW_LandMap.png') #Write the result to the present working directory as a PNG image file.
    
    assert True


def test_basemap_square():
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map = map_canvas.MapCanvas((300, 300), land_polygons=set)
    map.draw_background()
    map.save_background(os.path.join(basedir,"background_square.png"))
    assert True

def test_basemap_square2():
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map = map_canvas.MapCanvas((100, 300), set)
    map.draw_background()
    map.save_background(os.path.join(basedir,"background_square2.png"))
    assert True

def test_basemap_square3():
    map = map_canvas.MapCanvas((300, 100))
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    p1 = gnome.utilities.geometry.polygons.Polygon([[0,45],[1,45],[1,46],[0,46]], metadata=('name','land','1'))
    set = gnome.utilities.geometry.polygons.PolygonSet()
    set.append(p1)
    map = map_canvas.MapCanvas((300, 100), land_polygons=set)
    map.draw_background()
    map.save_background(os.path.join(basedir,"background_square3.png"))
    assert True


def test_basemap():
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    map = map_canvas.MapCanvas((400, 800), polygons)
    map.draw_background()
    map.save_background(os.path.join(basedir,"background1.png"))
    assert True

def test_basemap_wide():
    polygons = haz_files.ReadBNA(os.path.join(datadir, 'MapBounds_2Spillable2Islands2Lakes.bna'), "PolygonSet")
    map = map_canvas.MapCanvas((800, 400), polygons)
    map.draw_background()
    map.save_background(os.path.join(basedir,"background2.png"))
    
    assert True

