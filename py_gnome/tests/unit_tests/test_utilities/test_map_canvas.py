"""
Tests of the map canvas code.
"""

import os

from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome.utilities.geometry.polygons import Polygon, PolygonSet

basedir = os.path.dirname(__file__)
datadir = os.path.join(basedir, r"../sample_data")

input_file = os.path.join(datadir,
                          'MapBounds_2Spillable2Islands2Lakes.bna')


def test_render():
    """
    tests the ability to render a basemap from a bna
    """

    map_canvas.make_map(input_file, os.path.join(basedir,
                        'map_sample.png'))

    assert True


def test_render_BW():
    '''
    Test the creation of a black and white map with an island inset.

    note: it is rendered "almost black" on black...
    '''

    polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

    m = map_canvas.BW_MapCanvas((500, 500), land_polygons=polygons)
    m.draw_background()

    # Write the result to the present working directory as a PNG image file.
    # m.save_background('BW_LandMap.png')

    assert True


def test_basemap_square():
    p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
                 , 'land', '1'))

    poly_set = PolygonSet()
    poly_set.append(p1)

    gmap = map_canvas.MapCanvas((300, 300), land_polygons=poly_set)
    gmap.draw_background()
    gmap.save_background(os.path.join(basedir, 'background_square.png'))
    assert True


def test_basemap_square2():
    p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
                 , 'land', '1'))

    poly_set = PolygonSet()
    poly_set.append(p1)

    gmap = map_canvas.MapCanvas((100, 300), poly_set)
    gmap.draw_background()
    gmap.save_background(os.path.join(basedir, 'background_square2.png'
                         ))
    assert True


def test_basemap_square3():
    p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
                 , 'land', '1'))

    poly_set = PolygonSet()
    poly_set.append(p1)

    gmap = map_canvas.MapCanvas((300, 100), land_polygons=poly_set)
    gmap.draw_background()
    gmap.save_background(os.path.join(basedir, 'background_square3.png'
                         ))
    assert True


def test_basemap():
    polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

    gmap = map_canvas.MapCanvas((400, 800), polygons)
    gmap.draw_background()
    gmap.save_background(os.path.join(basedir, 'background1.png'))
    assert True


def test_basemap_wide():
    polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

    gmap = map_canvas.MapCanvas((800, 400), polygons)
    gmap.draw_background()
    gmap.save_background(os.path.join(basedir, 'background2.png'))

    assert True

def test_draw_raster_map():
    """
    tests drawing the raster map
    """
    import gnome
    polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

    gmap = map_canvas.MapCanvas((1000, 1000), polygons)
    gmap.viewport = ((-127.47,48.10),(-127.22, 48.24))
    gmap.draw_background()

    # make a raster map out of the BNA:
    raster_map = gnome.map.MapFromBNA(input_file, raster_size=10000)

    gmap.draw_raster_map(raster_map, outline=True)

    gmap.save_background(os.path.join(basedir, 'raster.png'))



