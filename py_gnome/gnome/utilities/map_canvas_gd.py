#!/usr/bin/env python
"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for rendering to images in scripting mode, and also for
pushing some rendering to the server.

Also used for making raster maps.

This should have the basic drawing stuff. Ideally nothig in here is
GNOME-specific.

This version used libgd and py_gd instead of PIL for the rendering
"""

import numpy as np

import py_gd

from gnome.utilities import projections

class MapCanvas(object):
    """
    A class to draw maps, etc.

    This class provides the ability to set a projection, and then
    change the viewport of the rendered map

    All the drawing methods will project and scale and shift the points so the
    rendered image is projected and shows only what is in the viewport.

    In addition, it keeps two image buffers: background and foreground. These
    can be rendered individually, and saved out either alone or composited.

    This version uses a paletted (8 bit) image -- may be updated for RGB images
    at some point.
    """

    def __init__(self,
                 image_size,
                 projection = projections.FlatEarthProjection(),
                 viewport=None,
                 preset_colors = 'BW',
                 background_color = 'transparent',
                 colordepth = 8,
                 ):
        """
        create a new map image from scratch -- specifying the size:
        Only the "Palette" image mode to used for drawing image.

        :param image_size: (width, height) tuple of the image size in pixels

        Optional parameters (kwargs)
        :param projection: gnome.utilities.projections object to use.
                           Default is
                           gnome.utilities.projections.FlatEarthProjection()

        :param viewport: viewport of map -- what gets drawn and on what
                         scale. Default is full globe: (((-180, -90), (180, 90)))

        :param preset_colors='BW': color set to preset. Options are:

                                      'BW' - transparent, black, and white: transparent background

                                      'web' - the basic named colors for the web: transparent background

                                      'transparent' - transparent background, no other colors set

                                      None - no pre-allocated colors -- the first one you allocate will
                                             be the background color

        :param background_color = 'transparent': color for the background -- must be a color that exists.

        :param colordepth=8: only 8 bit color supported for now
                             maybe someday, 32 bit will be an option
        """
        self._image_size = image_size

        if colordepth != 8:
            raise NotImplementedError("only 8 bit color currently implemented")
        self.background_color = background_color
        self.create_images(preset_colors)

        if viewport is None:
            viewport = ((-180, -90), (180, 90))

        projection.set_scale(viewport, self.image_size)
        self.projection = projection

    def viewport_to_dict(self):
        '''
        convert numpy arrays to list of tuples
        todo: this happens in multiple places so maybe worthwhile to define
        custom serialize/deserialize -- but do this for now
        '''
        return [tuple(i) for i in self.viewport]

    @property
    def viewport(self):
        """
        returns the current value of viewport of map:
        the bounding box of the image
        """
        return self.projection.to_lonlat(((0, self.image_size[1]),
                (self.image_size[0], 0)))

    @viewport.setter
    def viewport(self, viewport_BB):
        """
        Sets the viewport of the map: what gets drawn at what scale

        :param viewport_BB: the new viewport, as a BBox object, or in the form:
                            ( (min_long, min_lat),
                              (max_long, max_lat) )
        Images are cleared when this is changed
        """
        self.projection.set_scale(viewport_BB, self.image_size)
        self.back_image.clear()
        self.fore_image.clear()

    @property
    def image_size(self):
        """
        makes it read-only

        or we can add a setter to re-size the images if we need that
        """
        return self._image_size

    def add_colors(self, color_list):
        """
        Add a list of colors to the pallette

        :param color_list: list of colors - each elemnt of the list is a 2-tuple:
                           ('color_name', (r,g,b))
        """
        self.fore_image.add_colors(color_list)
        self.back_image.add_colors(color_list)

    def get_color_names(self):
        """
        returns all the names colors
        """
        return self.fore_image.get_color_names()

    def create_images(self, preset_colors):
        self.fore_image = py_gd.Image(width=self.image_size[0],
                                      height=self.image_size[1],
                                      preset_colors=preset_colors)

        self.back_image = py_gd.Image(width=self.image_size[0],
                                      height=self.image_size[1],
                                      preset_colors=preset_colors)
        self.clear_background()
        self.clear_foreground()

    def clear_background(self):
        self.back_image.clear(self.background_color)

    def clear_foreground(self):
        self.fore_image.clear('transparent')

    def copy_back_to_fore(self):
        """
        copy the background to the foreground

        note: this will write over anything on the foreground image
        """
        self.fore_image.copy(self.back_image, (0,0), (0,0), self.back_image.size)


    def draw_points(self,
                    points,
                    diameter=1,
                    color='black',
                    shape="round",
                    background=False):
        """
        Draws a set of individual points all in the same color

        :param points: a Nx2 numpy array, or something that can be turned in to one

        :param diameter=1: diameter of the points in pixels.
        :type diameter: integer

        :param color: a named color.
        :type color: string

        :param shape: what shape to draw, options are "round", "x".
        :type shape: string

        :param background=False: whether to draw to the background image.
        :type background: bool
        """
        if shape not in ['round', 'x']:
            raise ValueError('only "round" and "x" are supported shapes')

        points = self.projection.to_pixel(points, asint=True)

        img = self.back_image if background else self.fore_image
        if shape == 'round':
            img.draw_dots(points, diameter=diameter, color=color)
        elif shape == 'x':
            img.draw_xes(points, diameter=diameter, color=color)

    def draw_polygon(self,
                     points,
                     line_color=None,
                     fill_color=None,
                     line_width=1,
                     background=False):
        """
        Draw a polygon

        :param points: sequence of points
        :type points: Nx2 array of integers (or something that can be turned into one)

        :param line_color=None: the color of the outline
        :type line_color=None:  color name (string) or index (int)

        :param fill_color=None: the color of the filled polygon
        :type  fill_color: color name (string) or index (int)

        :param line_width=1: width of line
        :type line_width: integer

        :param background=False: whether to draw to the background image.
        :type background: bool


        """
        points = self.projection.to_pixel_2D(points, asint=True)
        img = self.back_image if background else self.fore_image

        img.draw_polygon(points,
                         line_color=line_color,
                         fill_color=fill_color,
                         line_width=line_width,
                         )

    def draw_polyline(self,
                      points,
                      line_color,
                      line_width=1,
                      background=False):
        """
        Draw a polyline

        :param points: sequence of points
        :type points: Nx2 array of integers (or somethign that can be turned into one)

        :param line_color: the color of the outline
        :type line_color:  color name or index

        :param line_width=1: width of line
        :type line_width: integer

        :param background=False: whether to draw to the background image.
        :type background: bool


        """
        points = self.projection.to_pixel_2D(points, asint=True)
        img = self.back_image if background else self.fore_image

        img.draw_polyline(points,
                         line_color=line_color,
                         line_width=line_width,
                         )

    def save_background(self, filename, file_type='png'):
        self.back_image.save(filename, file_type)

    def save_foreground(self, filename, file_type='png'):
        """
        save the foreground image to the specified filename

        :param filename: full path of file to be saved to
        :type filename: string

        :param draw_back_to_fore=True: whether to add the background image to the
                                       foreground before saving.
        :type draw_back_to_fore: bool

        :param file_type: type of file to save: options are: 'png', 'gif', 'jpeg', 'bmp'
        :type file_type: string
        """

        self.fore_image.save(filename, file_type = file_type)


# if __name__ == "__main__":
##    # a small sample for testing:
##    bb = np.array(((-30, 45), (-20, 55)), dtype=np.float64)
##    im = (100, 200)
##    proj = simple_projection(bounding_box=bb, image_size=im)
##    print proj.ToPixel((-20, 45))
##    print proj.ToLatLon(( 50., 100.))
#
#    bna_filename = sys.argv[1]
#    png_filename = bna_filename.rsplit(".")[0] + ".png"
#    bna = make_map(bna_filename, png_filename)
