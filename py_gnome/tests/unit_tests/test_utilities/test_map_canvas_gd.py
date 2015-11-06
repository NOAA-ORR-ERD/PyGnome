"""
Tests of the gd map canvas code.
"""

import os


from gnome.utilities import map_canvas_gd
from gnome.utilities.map_canvas_gd import MapCanvas
from gnome.utilities.file_tools import haz_files
from gnome.utilities.geometry.polygons import Polygon, PolygonSet

import pytest
from ..conftest import testdata, isclose


input_file = testdata["Renderer"]["bna_sample"]

def test_init():
    """
    can we even initialize one
    """
    mc = MapCanvas( (400,300) )

def test_set_colors():

    mc = MapCanvas( (400,300) )

    colors = [ ('blue', (  0,   0, 255)),
               ('red', (255,   0,   0))]

    mc.add_colors(colors)

    assert mc.fore_image.get_color_names() == mc.back_image.get_color_names()

    colors = mc.get_color_names()

    assert colors == ['transparent', 'black', 'white', 'blue', 'red']

def test_background_poly(output_dir):
    """
    test drawing polygons to the background
    """
    mc = MapCanvas( (400,300), preset_colors='web' )

    mc.background_color = 'transparent'
    mc.clear_background()

    mc.draw_polygon( ( (-30, 30),
                       ( 30, 30),
                       ( 30,-20),
                       (-30,-20)
                      ),
                      fill_color = 'blue',
                      line_color = 'black',
                      background = True,
                    )

    mc.save_background(os.path.join(output_dir, "background.png"))

def test_foreground_poly(output_dir):
    """
    test drawing polygons to the background
    """
    mc = MapCanvas( (400,300), preset_colors='web' )

    mc.background_color = 'transparent'
    mc.clear_background()

    mc.draw_polygon( ( (-30, 30),
                       ( 30, 30),
                       ( 30,-20),
                       (-30,-20)
                      ),
                      fill_color = 'white',
                      line_color = 'black',
                      background = False,
                    )

    mc.save_foreground(os.path.join(output_dir, "foreground.png"))

def test_foreground_polyline(output_dir):
    """
    test drawing polygons to the background
    """
    mc = MapCanvas( (400,300), preset_colors='web' )

    mc.background_color = 'transparent'
    mc.clear_background()

    mc.draw_polyline( ( (-30, 30),
                       ( 30, 30),
                       ( 30,-20),
                       (-30,-20)
                      ),
                      line_color = 'red',
                      line_width = 5,
                      background = False,
                    )

    mc.save_foreground(os.path.join(output_dir, "foreground_polyline.png"))

def test_copy_back_to_fore(output_dir):

    mc = MapCanvas( (400,300), preset_colors='web' )

    mc.background_color = 'white'
    mc.clear_background()
    mc.clear_foreground()

    mc.draw_polyline( ( (-30, 30),
                       ( 30, 30),
                       ( 30,-20),
                       (-30,-20)
                      ),
                      line_color = 'red',
                      line_width = 5,
                      background = True,
                    )
    mc.copy_back_to_fore()
    mc.draw_polyline( ( (-20, 20),
                       ( 20, 20),
                       ( 20,-10),
                       (-20,-10)
                      ),
                      line_color = 'blue',
                      line_width = 5,
                      background = False,
                    )

    mc.save_foreground(os.path.join(output_dir, "copy_back_to_fore.png"))


def test_projection(output_dir):
    """
    draw the "same sized" rectangle at three latitudes to see how the look
    """
    mc = MapCanvas( (400,400) , preset_colors='web')


    mc.viewport = ((-20,25),(20,65))
    mc.draw_polygon( ( (-15, 60),
                       ( 15, 60),
                       ( 15, 30),
                       (-15, 30)
                      ),
                      fill_color = 'maroon',
                      line_color = 'black',
                    )

    mc.save_foreground(os.path.join(output_dir, "image_projection_north.png"))

    mc.viewport = ((-20,-20),(20,20))
    mc.draw_polygon( ( (-15, 15),
                       ( 15, 15),
                       ( 15, -15),
                       (-15, -15)
                      ),
                      fill_color = 'maroon',
                      line_color = 'black',
                    )

    mc.save_foreground(os.path.join(output_dir, "image_projection_equator.png"))

    mc.viewport = ((-20,-45),(20,-90))
    mc.draw_polygon( ( (-15, -80),
                       ( 15, -80),
                       ( 15, -50),
                       (-15, -50)
                      ),
                      fill_color = 'maroon',
                      line_color = 'black',
                    )

    mc.save_foreground(os.path.join(output_dir, "image_projection_south.png"))

@pytest.mark.parametrize( ("val","exp","num_digits"), [(    1.1,     1.0,      1 ),
                                                       ( 0.0011,     0.001,    1 ),
                                                       (  1100.,     1000.0,   1 ),
                                                       (  1900 ,     2000.0,   1 ),
                                                       (  0.0000154, 0.00002,  1 ),
                                                       (  0.0000154, 0.000015, 2 ),
                                                       (  1234.56,   1200.0,   2 ),
                                                       (  1234567,   1230000,      3 ),
                                                      ]
                         )
def test_round_to_digit(val, exp, num_digits):

    rounded = MapCanvas._round_to_digit(val, num_digits)

    assert isclose(rounded, exp, rel_tol=1e-14)




# def test_render(dump):
#     """
#     tests the ability to render a basemap from a bna
#     """

#     map_canvas.make_map(input_file, os.path.join(dump, 'map_sample.png'))

#     assert True


# def test_render_BW():
#     '''
#     Test the creation of a black and white map with an island inset.

#     note: it is rendered "almost black" on black...
#     '''

#     polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

#     m = map_canvas.BW_MapCanvas((500, 500), land_polygons=polygons)
#     m.draw_background()

#     # Write the result to the present working directory as a PNG image file.
#     # m.save_background('BW_LandMap.png')

#     assert True


# def test_basemap_square(dump):
#     p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
#                  , 'land', '1'))

#     poly_set = PolygonSet()
#     poly_set.append(p1)

#     gmap = map_canvas.MapCanvas((300, 300), land_polygons=poly_set)
#     gmap.draw_background()
#     gmap.save_background(os.path.join(dump, 'background_square.png'))
#     assert True


# def test_basemap_square2(dump):
#     p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
#                  , 'land', '1'))

#     poly_set = PolygonSet()
#     poly_set.append(p1)

#     gmap = map_canvas.MapCanvas((100, 300), poly_set)
#     gmap.draw_background()
#     gmap.save_background(os.path.join(dump, 'background_square2.png'
#                          ))
#     assert True


# def test_basemap_square3(dump):
#     p1 = Polygon([[0, 45], [1, 45], [1, 46], [0, 46]], metadata=('name'
#                  , 'land', '1'))

#     poly_set = PolygonSet()
#     poly_set.append(p1)

#     gmap = map_canvas.MapCanvas((300, 100), land_polygons=poly_set)
#     gmap.draw_background()
#     gmap.save_background(os.path.join(dump, 'background_square3.png'))
#     assert True


# def test_basemap(dump):
#     polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

#     gmap = map_canvas.MapCanvas((400, 800), polygons)
#     gmap.draw_background()
#     gmap.save_background(os.path.join(dump, 'background1.png'))
#     assert True


# def test_basemap_wide(dump):
#     polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

#     gmap = map_canvas.MapCanvas((800, 400), polygons)
#     gmap.draw_background()
#     gmap.save_background(os.path.join(dump, 'background2.png'))

#     assert True


# def test_draw_raster_map(dump):
#     """
#     tests drawing the raster map
#     """
#     import gnome
#     polygons = haz_files.ReadBNA(input_file, 'PolygonSet')

#     gmap = map_canvas.MapCanvas((1000, 1000), polygons)
#     gmap.viewport = ((-127.47,48.10),(-127.22, 48.24))
#     gmap.draw_background()

#     # make a raster map out of the BNA:
#     raster_map = gnome.map.MapFromBNA(input_file, raster_size=10000)

#     gmap.draw_raster_map(raster_map, outline=True)

#     gmap.save_background(os.path.join(dump, 'raster.png'))
