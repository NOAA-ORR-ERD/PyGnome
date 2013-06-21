#!/usr/bin/env python

"""
unit tests for the py_gd project

designed to be run with pytest:

py.test test_gd.py

"""


import pytest

import numpy as np

import py_gd


def test_init():
    img = py_gd.Image(width=400, height=400)

    img = py_gd.Image(400, 400)

    ## need to pass in width and height
    with pytest.raises(TypeError):
        py_gd.Image()
    with pytest.raises(TypeError):
        py_gd.Image(200)

def test_mem_limit():
    """
    test the limit for largest image.

    note that the 1GB max is arbitrary -- youc an change it iin the code.

    But my system, at least, will try to allocate much more memory that
    you'd want, bringing the system to an almost halt, before raising 
    a memory error, so I sete a limit.
    """
    img = py_gd.Image(32768, 32768) # 1 GB image

    with pytest.raises(MemoryError):
        img = py_gd.Image(32768, 32769) # > 1 GB image


def test_info():
    img = py_gd.Image(400, 300)

    assert  str(img) == "py_gd.Image: width:400 and height:300"

    assert repr(img) == "Image(width=400, height=300)"

def test_add_colors():
    img = py_gd.Image(10, 10, preset_colors='basic')

    assert img.get_color_names() == ['transparent', 'black', 'white']

    img.add_color('light grey', (220,220,220) )
    assert img.get_color_names() == ['transparent', 'black', 'white', 'light grey']
    assert img._get_color_index('light grey') == 3

    img.draw_rectangle((2,2), (7,7), fill_color='light grey')
    img.save('test_image_color.bmp')

    with pytest.raises(ValueError):
        # color doesn't exist
        img.draw_rectangle((2,2), (7,7), fill_color='red')

def test_add_colors_repeat():
    img = py_gd.Image(10, 10, preset_colors='basic')

    index_1 = img.add_color('blue', (0, 0, 255))
    
    with pytest.raises(ValueError):
        # adding one with the same name should raise an exception
        img.add_color('blue', (0, 0, 200))

    # Adding the same color with a different name:
    # adds another index -- should it?
    index_2 = img.add_color('full_blue', (0, 0, 255))
    assert index_1 != index_2


def test_add_colors_max():
    img = py_gd.Image(10, 10, preset_colors='basic')

    # should be able to add this many:
    for i in range(253):
        img.add_color("color_%i"%i, (i, i, i) )

    # adding one more should raise an exception:
    with pytest.raises(ValueError):
        img.add_color("color_max", (10, 100, 200) )


def test_save_image():
    img = py_gd.Image(400, 300)

    img.draw_line( (0,   0), (399, 299), 'white', line_width=4)
    img.draw_line( (0, 299), (399, 0), 'green', line_width=4)
    
    img.save("test_image_save.bmp")

    img.save("test_image_save.jpg", "jpeg")

    img.save("test_image_save.gif", "gif")

    img.save("test_image_save.png", "png")

    with pytest.raises(ValueError):
        img.save("test_image1.something", "random_string")

def test_line():
    img = py_gd.Image(100,200)

    img.draw_line( (0, 0), (99, 199), 'white')
    img.draw_line( (0, 199), (99, 0), 'red', line_width=2)
    img.draw_line( (0, 100), (99, 100), 'green', line_width=4)
    img.draw_line( (50, 0), (50, 199), 'blue', line_width=8)
    img.save("test_image_line.bmp")

    with pytest.raises(TypeError):
        img.draw_line( (0, 0), (99, 199), 'white', line_width='fred')


def test_line_clip():
    img = py_gd.Image(100,200)

    img.draw_line( (-30, -10), (150, 250), 'white')
    img.save("test_image_line_clip.bmp")


def test_SetPixel():
    img = py_gd.Image(5,5)

    img.draw_pixel( (0, 0), 'white')
    img.draw_pixel( (1, 1), 'red')
    img.draw_pixel( (2, 2), 'green')
    img.draw_pixel( (3, 3), 'blue')

    img.save("test_image_pixel.bmp")

def test_GetPixel():
    img = py_gd.Image(5,5)

    img.draw_pixel( (0, 0), 'white')
    img.draw_pixel( (1, 1), 'red')
    img.draw_pixel( (2, 2), 'green')
    img.draw_pixel( (3, 3), 'blue')

    assert img.get_pixel_color( (0, 0) ) == 'white' 
    assert img.get_pixel_color( (1, 1) ) == 'red'
    assert img.get_pixel_color( (2, 2) ) == 'green'
    assert img.get_pixel_color( (3, 3) ) == 'blue'

def test_Polygon1():
    img = py_gd.Image(100,200)

    points = ( ( 10,  10),
               ( 20, 190),
               ( 90,  10),
               ( 50,  50),
              )

    img.draw_polygon( points, 'red')
    img.save("test_image_poly1.bmp")

def test_Polygon2():
    img = py_gd.Image(100,200)

    points = ( ( 10,  10),
               ( 20, 190),
               ( 90,  10),
               ( 50,  50),
              )

    img.draw_polygon( points, fill_color='blue')
    img.save("test_image_poly2.bmp")

def test_Polygon3():
    img = py_gd.Image(100,200)

    points = ( ( 10,  10),
               ( 20, 190),
               ( 90,  10),
               ( 50,  50),
              )

    img.draw_polygon( points, fill_color='blue',line_color='red', line_width=4)
    img.save("test_image_poly3.bmp")

def test_polygon_clip():
    img = py_gd.Image(100,200)

    img = py_gd.Image(100,200)

    points = ( ( -20,  10),
               ( 20, 250),
               ( 120,  10),
               ( 50,  50),
              )

    img.draw_polygon( points, fill_color='blue',line_color='red')
    img.save("test_image_polygon_clip.bmp")

def test_polyline():
    img = py_gd.Image(100,200)

    points = ( ( 10,  10),
               ( 20, 190),
               ( 90,  10),
               ( 50,  50),
              )

    img.draw_polyline( points, 'red', line_width=3)

    points = ( ( 50,  50),
               ( 90, 190),
               ( 10,  10),
              )

    img.draw_polyline( points, 'blue', line_width=5)

    with pytest.raises(ValueError):
        img.draw_polyline( ((10,10),(90,90)), 'blue')

    img.save("test_image_polyline.bmp")


def test_rectangles():
    img = py_gd.Image(100,200)

    img.draw_rectangle( (10,10), (30,40), fill_color='blue')
    img.draw_rectangle( (20,50), (40,70), line_color='blue', line_width=5)
    img.draw_rectangle( (40,80), (90,220), fill_color='white',line_color='green', line_width=2)
    img.save("test_image_rectangle.bmp")

def test_arc():
    img = py_gd.Image(400,600)
    # possible flags:  "Arc", "Pie", "Chord", "NoFill", "Edged" (Arc and Pie are the same)
    center = (200, 150)
    # just the lines

    img.draw_arc( center, 380, 280, start=-30, end= 30, line_color='white', style='arc',   draw_wedge=False)
    img.draw_arc( center, 380, 280, start= 30, end= 90, line_color='white', style='chord', draw_wedge=False, line_width=3)
    img.draw_arc( center, 380, 280, start= 90, end=150, line_color='white', style='arc',   draw_wedge=True, line_width=5)
    img.draw_arc( center, 380, 280, start=150, end=210, line_color='white', style='chord', draw_wedge=True)

    # just fill
    img.draw_arc( center, 380, 280, start=210, end= 270, fill_color='purple', style='arc')
    img.draw_arc( center, 380, 280, start=270, end= 330, fill_color='teal', style='chord')

    # line and fill
    center = (200, 450)

    img.draw_arc( center, 380, 280, start= 30, end= 90, line_color='white', fill_color='green', style='chord')
    #img.draw_arc( center, 380, 280, start= 90, end= 150, line_color='white', fill_color='blue', styles=['NoFill'])
    img.draw_arc( center, 380, 280, start=150, end= 210, line_color='green', fill_color='white', style='arc')
    # img.draw_arc( center, 380, 280, start=210, end= 270, line_color='white', fill_color='purple', styles=['Chord','Edged', 'NoFill'])
    img.draw_arc( center, 380, 280, start=270, end= 330, line_color='blue', fill_color='red', line_width=3)

    img.save("test_image_arc.bmp")

    #errors
    with pytest.raises(ValueError):
        img.draw_arc( center, 380, 280, start= 30, end= 90, line_color='white', style='fred')


def test_text():
    img = py_gd.Image(200, 200)


    img.draw_text("Some Tiny Text", (20, 20), font="tiny", color='white')
    img.draw_text("Some Small Text", (20, 40), font="small", color='white')
    img.draw_text("Some Medium Text", (20, 60), font="medium", color='white')
    img.draw_text("Some Large Text", (20, 80), font="large", color='white')
    img.draw_text("Some Giant Text", (20, 100), font="giant", color='white')
    img.save("test_image_text.png", "png")



def test_colors():
    img = py_gd.Image(5, 5)

    # this shold work
    img._get_color_index('black')

    # so should this:
    img._get_color_index(0)
    img._get_color_index(255)

    # will round floating point numbers
    # shoul dthi sbe changed?
    assert img._get_color_index(2.3) == 2

    with pytest.raises(ValueError):
        # error if index not in 0--255
        img._get_color_index(300)

    with pytest.raises(ValueError):
        # error if color is not in dict
        img._get_color_index('something else')

    with pytest.raises(ValueError):
        # error if color is not anumber
        img._get_color_index((1,2,3))

    with pytest.raises(TypeError):
        # error if color is unhasable
        img._get_color_index(['a', 'random', 4])



def test_array():
    img = py_gd.Image(10, 5)
    img.draw_line( (0, 0), (9, 4), 'black', line_width=1)
    print "result from __array__", img.__array__()
    arr = np.asarray(img)
    assert np.array_equal(arr, [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
                                )

def test_array_set():
    arr = np.array( [[ 0, 1, 2],
                     [ 3, 4, 0],
                     [ 1, 2, 3],
                     [ 4, 0, 1] ],
                     dtype=np.uint8)
 
    img = py_gd.Image(arr.shape[1], arr.shape[0])
    img.set_data(arr)
    
    img.save('junk.bmp')

    for y in range(img.height):
        for x in range(img.width):
            assert arr[y,x] == img.get_pixel_value( (x,y) )

def test_array_creation():
    arr = np.array( [[ 0, 1, 2],
                     [ 3, 4, 0],
                     [ 1, 2, 3],
                     [ 4, 0, 1] ],
                     dtype=np.uint8)
 
    img = py_gd.from_array(arr)
    
    img.save('junk.bmp')

    for y in range(img.height):
        for x in range(img.width):
            assert arr[y,x] == img.get_pixel_value( (x,y) )


