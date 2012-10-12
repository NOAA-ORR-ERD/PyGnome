"""
Tests of the map canvas code.
"""

import pytest

import numpy as np

from gnome import basic_types
from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files

def test_render_RGB():
    map_canvas.make_map("SampleData/MapBounds_2Spillable2Islands2Lakes.bna", "junk_rgb.png", format='RGB')

    assert True

def test_render_Pallette():
    map_canvas.make_map("SampleData/MapBounds_2Spillable2Islands2Lakes.bna", "junk_p.png", format='palette')

    assert True

def test_basemap():
    map = map_canvas.Palette_MapCanvas((800, 600))
    polygons = haz_files.ReadBNA("SampleData/MapBounds_2Spillable2Islands2Lakes.bna", "PolygonSet")
    map.draw_land(polygons)
    map.save("junk.png")
    

    


