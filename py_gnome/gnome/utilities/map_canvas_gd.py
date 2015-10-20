#!/usr/bin/env python
# coding=utf8
"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for rendering to images in scripting mode, and also for
pushing some rendering to the server.

Also used for making raster maps.

This should have the basic drawing stuff. Ideally nothig in here is
GNOME-specific.

This version used libgd and py_gd instead of PIL for the rendering
"""

from math import floor, log10

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
                 projection = None,
                 viewport=None,
                 preset_colors = 'BW',
                 background_color = 'transparent',
                 colordepth = 8,
                 ):
        """
        create a new map image from scratch -- specifying the size

        :param image_size: (width, height) tuple of the image size in pixels

        Optional parameters

        :param projection=None: gnome.utilities.projections object to use.
                                if None, it defaults to FlatEarthProjection()

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

        projection = projections.FlatEarthProjection() if projection is None else projection
        self._image_size = image_size

        if colordepth != 8:
            raise NotImplementedError("only 8 bit color currently implemented")

        self.background_color = background_color
        self.create_images(preset_colors)
        self.projection = projection

        if viewport is None:
            self._viewport = ((-180, -90), (180, 90))
        else: self._viewport = viewport

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def viewport_to_dict(self):
        '''
        convert numpy arrays to list of tuples
        todo: this happens in multiple places so maybe worthwhile to define
        custom serialize/deserialize -- but do this for now
        '''
        return map(tuple, self._viewport.tolist())

    @property
    def viewport(self):
        """
        returns the current value of viewport of map:
        the bounding box of the image
        """
        return self._viewport.BB

    @viewport.setter
    def viewport(self, BB):
        """
        viewport setter for bounding box only...allows map_canvas.viewport = ((x1,y1),(x2,y2))
        """
        self._viewport.BB = BB if BB else self._viewport.BB
        
    def set_viewport(self, BB = None, center = None, width = None, height = None):
        """
        Function to allow the user to set properties of the viewport in meters, or by bounding box
        :param center: The point around which the viewport is centered
        :type a tuple containing an x/y coordinate

        :param width: Width of the viewport in meters

        :param height: height of the viewport in meters

        :param BB: Bounding box of the viewport (overrides all previous parameters)
        :type a list of tuples containing of the lower left and top right coordinates
        
        """
        if BB is None:
            self._viewport.center = center
            distances = self.projection.meters_to_lonlat((width, height, 0), (center[0], center[1],0))
            self._viewport.width = distances[0]
            self._viewport.height = distances[1]
        else:
            self._viewport.BB = BB
            
        self.rescale()

    def zoom(self, multiplier):
        self._viewport.scale(multiplier)
        self.rescale()

    def rescale(self):
        """
        Rescales the projection to the viewport bounding box. Should be called whenever the viewport changes
        """
        self.projection.set_scale(self._viewport.BB, self.image_size)
        self.back_image.clear()
        self.fore_image.clear()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        if preset_colors is not None:
            "can't clear image if there are no colors"
            self.clear_background()
            self.clear_foreground()

    def back_asarray(self):
        """
        return the background image as a numpy array
        """
        return np.asarray(self.back_image)

    def fore_asarray(self):
        """
        return the foreground image as a numpy array
        """
        return np.asarray(self.fore_image)

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

    def draw_graticule(self, background=False):
        """
        draw a graticule (grid lines) on the map

        only supports decimal degrees for now...
        """
        ( (min_lat, min_lon), (max_lat, max_lon) ) = self.viewport

        d_lat = max_lat - min_lat
        d_lon = max_lon - min_lon

        # Want about one grid line per 100 pixels
        ppg = 100.0 # a float!

        delta_lon = d_lon / (self.image_size[0]/ppg)
        delta_lat = d_lat / (self.image_size[1]/ppg)

'''
 """
        Draws a lat/long grid onto the background. WHY does it spit out a blank image?
        """
        self.grid_image = Image.new(self.image_mode, self.image_size, color=self.colors['transparent'])
        self.grid_image.putpalette(self.palette)
        grid_pen = ImageDraw.Draw(self.grid_image)
        max_lat = self.map_BB[1][1]
        min_lat = self.map_BB[0][1]
        max_long = self.map_BB[1][0]
        min_long = self.map_BB[0][0]
        delta_lat = (max_lat - min_lat)/self.grid_lines
        delta_long = (max_long - min_long)/self.grid_lines
        cur_x = 0
        cur_y = 0
        for i in range(0,self.grid_lines):
            #grid_pen.line([(min_long,min_lat+delta_lat*i),(max_long,min_lat+delta_lat*i)], width=1)
            #grid_pen.line([(min_long+delta_long*i,min_lat),(min_long+delta_long*i,max_lat)], width=1)
            grid_pen.line([cur_x, 0, cur_x, self.image_size[1]], width=1)
            grid_pen.line([0, cur_y, self.image_size[0], cur_y], width=1)
            cur_x += self.image_size[0]/self.grid_lines
            cur_y += self.image_size[1]/self.grid_lines
            
        return None
'''

    @staticmethod
    def _find_graticule_locations(image_size, viewport,  units="decimal_degrees"):
        """
        finds where we want the graticlue located.

        intended to be used internally by the graticule printer

        """
        ( (min_lat, min_lon), (max_lat, max_lon) ) = viewport

        d_lat = max_lat - min_lat
        d_lon = max_lon - min_lon

        # Want about one grid line per 100 pixels
        ppg = 100.0 # a float!

        delta_lon = d_lon / (image_size[0]/ppg)
        delta_lat = d_lat / (image_size[1]/ppg)

        # round to single digit

        print delta_lon, delta_lat

        delta_lon, delta_lat = MapCanvas._round_to_digit(delta_lon, 2), MapCanvas._round_to_digit(delta_lat, 2)



    @staticmethod
    def _round_to_digit(value, num_digits = 1):
        """
        rounds to teh number significatn figures requested
        used by _find_graticule_locations

        :param value: the value to round

        :param num_digits: number of digits to preserve

        """
        mag = 10**floor( log10( value )-(num_digits-1))
        return round(value / mag) * mag


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

## Gridlines code borrowed from MapRoom
import bisect
class GridLines(object):
    DEGREE = np.float64(1.0)
    MINUTE = DEGREE / 60.0
    SECOND = MINUTE / 60.0

    DMS_STEPS = (
        MINUTE,
        MINUTE * 2,
        MINUTE * 3,
        MINUTE * 4,
        MINUTE * 5,
        MINUTE * 10,
        MINUTE * 15,
        MINUTE * 20,
        MINUTE * 30,
        DEGREE,
        DEGREE * 2,
        DEGREE * 3,
        DEGREE * 4,
        DEGREE * 5,
        DEGREE * 10,
        DEGREE * 15,
        DEGREE * 20,
        DEGREE * 30,
        DEGREE * 40,
    )
    DMS_COUNT = len(DMS_STEPS)
    
    DEGREE = np.float64(1.0)
    TENTH = DEGREE / 10.0
    HUNDREDTH = DEGREE / 100.0
    THOUSANDTH = DEGREE / 1000.0


    DEG_STEPS = (
        THOUSANDTH,
        THOUSANDTH * 2,
        THOUSANDTH * 5,
        HUNDREDTH,
        HUNDREDTH * 2,
        HUNDREDTH * 5,
        TENTH,
        TENTH * 2,
        TENTH * 5,
        DEGREE,
        DEGREE * 2,
        DEGREE * 3,
        DEGREE * 4,
        DEGREE * 5,
        DEGREE * 10,
        DEGREE * 15,
        DEGREE * 20,
        DEGREE * 30,
        DEGREE * 40,
    )
    DEG_COUNT = len(DEG_STEPS)
    
    def __init__(self, viewport=None, line_range=(8,8), DegMinSec=False):
        """
        Creates a GridLines instance that does the logic for and describes the current graticule

        :param viewport: bounding box of the viewport in question. 
        :type viewport: tuple of lon/lat

        :param line_range: How many lines to be displayed on the longest dimension of the viewport. Graticule will scale up
        or down only when the number of lines in the viewport falls outside the range.
        :type line_range: tuple of integers

        :param DegMinSec: Whether measurement is in Degrees/Minute/Seconds, or decimal lon/lat
        :type bool
        """
        self.viewport = Viewport()
        if viewport not None:
            self.viewport = viewport
            
        self.type = type
        if DegMinSec :
            self.STEPS = DMS_STEPS
            self.STEP_COUNT = DMS_COUNT:
        else:
            self.STEPS = DEG_STEPS
            self.STEP_COUNT = DEG_COUNT:
        
        self.num_drawn = line_avg = (line_range[1] + line_range[0])//2
        self.ref_dim = 'w' if viewport.width >= viewport.height) else 'h'
        self.ref_len = self.viewport.width if self.ref_dim is 'w' else self.viewport.height
        self.current_interval = get_step_size(ref_dim)
        
    """
    class to hold logic for determining where the gridlines should be
    for the graticule
    """
    def get_step_size(self, reference_size):
        """
        get the steps required given a reference_size

        :param reference_size: the approximate size you want
        """
        return self.STEPS[min( bisect.bisect(self.STEPS, abs(reference_size)),
                               self.STEP_COUNT - 1,)
                         ]
    
    def get_lines(self):
        min = self.viewport.BB[0,0] if self.ref_dim is 'w' else self.viewport.BB[0,1]
        max = self.viewport.BB[1,0] if self.ref_dim is 'w' else self.viewport.BB[1,1]
        start = (min//self.current_interval) * self.current_interval
        end = (max // self.current_interval + 1) * self.current_interval
        lines = [((x1,y1),(x2,y2)) ] 
        
    def format_lat_line_label(self, latitude):
        return coordinates.format_lat_line_label(latitude)

    def format_lon_line_label(self, longitude):
        return coordinates.format_lon_line_label(longitude)

    def format_lat_line_label(self, latitude):
        ( degrees, direction ) = \
            coordinates.float_to_degrees(latitude, directions=("N", "S"))

        return u" %.2f° %s " % (degrees, direction)

    def format_lon_line_label(self, longitude):
        ( degrees, direction ) = \
            coordinates.float_to_degrees(longitude, directions=("E", "W"))

        return u" %.2f° %s " % (degrees, direction)

class GridScale(object):
    """
    A GridScale class controls what the graticule looks like. It manages and serves a list of lines, that when printed, form the graticule.
    The list of lines is based on the size of the viewport. If the viewport changes, it should refresh this class, which will then 
    recompute new lines based on the new dimensions.
    
    
    """
    def __init__(self, image_size=None, viewport=None, delta_lonlat = None, lines_per_dim = None):

class Viewport(object):
    
    """
    Viewport

    class that defines and manages attribues for a viewport onto a flat 2D map. All points and measurements are in lon/lat
    

    """
    def __init__(self, BB = None, center=None, width = None, height = None):
        """
        Init the viewport. Can initialize with center/width/height, and/or with bounding box. 
        NOTE: Bounding box takes precedence over any previous parameters

        :param center: The point around which the viewport is centered
        :type a tuple containing an lon/lat coordinate

        :param width: Width of the viewport (lon)

        :param height: height of the viewport (lat)

        :param BB: Bounding box of the viewport (overrides previous parameters)
        :type a list of lon/lat tuples containing of the lower left and top right coordinates
        """
        self._BB = None
        self._center = None
        self._width = None
        self._height = None
        if BB is None:
            if center is None:
                raise ValueError("Center is unspecified")
            if width is None:
                raise ValueError("Width is unspecified")
            if height is None:
                raise ValueError("Height is unspecified")
    
            self._center = center
            self._width = width
            self._height = height
            self.recompute_BB()
        else:
            self._BB = BB
            self.recompute_dim()
                
    def scale(self, multiplier=1.0):
        self._width *= multiplier
        self._height *= multiplier
        self.recompute_BB()
        
    def recompute_dim(self):
        self._width = self.BB[1][0] - self.BB[0][0]
        self._height = self.BB[1][1] - self.BB[0][1] 
        self._center = (self.BB[1][0] - self.width/2.0,
                       self.BB[1][1] - self.height/2.0)
        
        
    def recompute_BB(self):
        halfx = self.width/2.0
        halfy = self.height/2.0
        self._BB = ((self.center[0] - halfx, self.center[0] - halfy),
                     (self.center[1] + halfx, self.center[1] + halfy)) 
        
    @property
    def BB(self):
        return self._BB
    
    @BB.setter
    def BB(self, BB):
        self._BB = BB if BB else self._BB
        self.recompute_dim()
        
    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, center):
        self._center = center if center else self._center
        self.recompute_BB()
        
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, width):
        self._width = width if width else self._width
        self.recompute_BB()
        
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, height):
        self._height = height if height else self._height
        self.recompute_BB()
        