#!/usr/bin/env python
# coding=utf8
"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for rendering to images in scripting mode, and also for
pushing some rendering to the server.

Also used for making raster maps.

This should have the basic drawing stuff. Ideally nothing in here is
GNOME-specific.
"""

import bisect

import numpy as np

import py_gd

import nucos as uc

from gnome.utilities.projections import FlatEarthProjection


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
                 image_size=(800, 600),
                 projection=None,
                 viewport=None,
                 preset_colors='BW',
                 background_color='transparent',
                 colordepth=8,
                 **kwargs):
        """
        create a new map image from scratch -- specifying the size

        :param image_size: (width, height) tuple of the image size in pixels

        Optional parameters

        :param projection=None: gnome.utilities.projections object to use.
                                if None, it defaults to FlatEarthProjection()

        :param viewport: viewport of map -- what gets drawn and on what
                         scale.
                         Default is full globe: (((-180, -90), (180, 90)))

        :param preset_colors='BW': color set to preset. Options are:

                                   'BW' - transparent, black, and white:
                                          transparent background

                                   'web' - the basic named colors for the web:
                                           transparent background

                                   'transparent' - transparent background,
                                                   no other colors set

                                   None - no pre-allocated colors
                                          -- the first one you allocate will
                                             be the background color

        :param background_color = 'transparent': color for the background
                                                 -- must be a color that exists

        :param colordepth=8: only 8 bit color supported for now
                             maybe someday, 32 bit will be an option
        """
        projection = (FlatEarthProjection()
                      if projection is None
                      else projection)
        self._image_size = image_size

        if colordepth != 8:
            raise NotImplementedError("only 8 bit color currently implemented")

        self.background_color = background_color
        self.create_images(preset_colors)
        self.projection = projection
        self._viewport = Viewport(((-180, -90), (180, 90)))

        if viewport is not None:
            self._viewport.BB = tuple(map(tuple, viewport))

        self.projection.set_scale(self.viewport, self.image_size)
        self.graticule = GridLines(self._viewport, self.projection)
        super(MapCanvas, self).__init__()

    def viewport_to_dict(self):
        '''
        convert numpy arrays to list of tuples
        todo: this happens in multiple places so maybe worthwhile to define
        custom serialize/deserialize -- but do this for now
        '''
        return list(map(tuple, self._viewport.BB))

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
        viewport setter for bounding box only
        - allows map_canvas.viewport = ((x1,y1),(x2,y2))
        """
        self._viewport.BB = BB if BB is not None else self.viewport.BB
        self.rescale()

    def set_viewport(self, BB=None, center=None, width=None, height=None):
        """
        Function to allow the user to set properties of the viewport in meters,
        or by bounding box

        :param center: The point around which the viewport is centered
        :type center: tuple of floats (lon, lat)

        :param width: Width of the viewport in meters
        :type width: float

        :param height: height of the viewport in meters
        :type height: float

        :param BB: Bounding box of the viewport
                   (overrides all previous parameters)

        :type BB: a list of tuples containing of the lower left and top right
              coordinates: ((min_x, min_y),(max_x, max_y))
        """

        if BB is None:
            self._viewport.center = center

            distances = (self.projection
                         .meters_to_lonlat((width, height, 0),
                                           (center[0], center[1], 0)))

            self._viewport.width = distances[0][0]
            self._viewport.height = distances[0][1]
        else:
            self._viewport.BB = BB

        self.rescale()

    def zoom(self, multiplier):
        self._viewport.scale(multiplier)
        self.rescale()

    def shift_viewport(self, delta):
        self._viewport.center = (self._viewport.center[0] + delta[0],
                                 self._viewport.center[1] + delta[1])
        self.rescale()

    def rescale(self):
        """
        Rescales the projection to the viewport bounding box.
        Should be called whenever the viewport changes
        """
        self.projection.set_scale(self.viewport, self.image_size)
        self.graticule.refresh_scale()

        self.clear_background()
        self.draw_background()

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

        :param color_list: list of colors. Each element of the list is a 2-tuple:
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
        width, height = self.image_size[:2]
        self.fore_image = py_gd.Image(width=width, height=height,
                                      preset_colors=preset_colors)

        self.back_image = py_gd.Image(width=width, height=height,
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
        self.fore_image.copy(self.back_image,
                             (0, 0), (0, 0),
                             self.back_image.size)

    def draw_points(self,
                    points,
                    diameter=1,
                    color='black',
                    shape="round",
                    background=False):
        """
        Draws a set of individual points all in the same color

        :param points: a Nx2 numpy array, or something that can be turned
                       into one

        :param diameter=1: diameter of the points in pixels.
        :type diameter: integer

        :param color: a named color.
        :type color: string

        :param shape: what shape to draw, options are "round", "x".
        :type shape: string

        :param background=False: whether to draw to the background image.
        :type background: bool
        """
        if shape not in ('round', 'x'):
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
        :type points: Nx2 array of integers (or something that can be turned
                      into one)

        :param line_color=None: the color of the outline
        :type line_color=None:  color name (string) or index (int)

        :param fill_color=None: the color of the filled polygon
        :type  fill_color: color name (string) or index (int)

        :param line_width=1: width of line
        :type line_width: integer

        :param background=False: whether to draw to the background image.
        :type background: bool
        """
        points = self.projection.to_pixel(points, asint=True)
        img = self.back_image if background else self.fore_image

        img.draw_polygon(points,
                         line_color=line_color,
                         fill_color=fill_color,
                         line_width=line_width)

    def draw_polyline(self,
                      points,
                      line_color,
                      line_width=1,
                      background=False):
        """
        Draw a polyline

        :param points: sequence of points
        :type points: Nx2 array of integers (or something that can be turned
                      into one)

        :param line_color: the color of the outline
        :type line_color:  color name or index

        :param line_width=1: width of line
        :type line_width: integer

        :param background=False: whether to draw to the background image.
        :type background: bool
        """
        points = self.projection.to_pixel(points, asint=True)

        img = self.back_image if background else self.fore_image

        img.draw_polyline(points,
                          line_color=line_color,
                          line_width=line_width)

    def draw_text(self, text_list, size='small', color='black', align='lt',
                  background=None, draw_to_back=False):
        """
        Draw ascii text to the image

        :param text_list: sequence of strings to be printed, and the locations
                          they are to be located
        :type text_list: ['string', (lon, lat)]

        :param size: size of the text to be printed
        :type size: one of the following strings:
                    {'tiny', 'small', 'medium', 'large', 'giant'}

        :param color: color of the text to be printed
        :type color: a valid color string in the py_gd Image color palettes

        :param align: sets the principal point of the text bounding box.
        :type align: one of the following strings:
                     {'lt', 'ct', 'rt', 'r', 'rb', 'cb', 'lb', 'l'}
        """
        img = self.back_image if draw_to_back else self.fore_image

        for tag in text_list:
            point = np.array((tag[1][0], tag[1][1])).reshape(-1, 2)
            point = self.projection.to_pixel(point, asint=True)[0]

            img.draw_text(tag[0], point, size, color, align, background)

    def draw_background(self):
        self.clear_background()
        self.draw_graticule()
        self.draw_tags()
        self.draw_grid()

    def draw_land(self):
        return None

    def draw_graticule(self, background=True):
        """
        draw a graticule (grid lines) on the map

        only supports decimal degrees for now...
        """
        for line in self.graticule.get_lines():
            self.draw_polyline(line, 'black', 1, background)

    def draw_grid(self):
        # Not Implemeneted in MapCanvas
        return None

    def draw_tags(self, draw_to_back=True):
        self.draw_text(self.graticule.get_tags(), draw_to_back=draw_to_back)

    def save_background(self, filename, file_type='png'):
        self.back_image.save(filename, file_type)

    def save_foreground(self, filename, file_type='png'):
        """
        save the foreground image to the specified filename

        :param filename: full path of file to be saved to
        :type filename: string

        :param file_type: type of file to save
        :type file_type: one of the following:
                         {'png', 'gif', 'jpeg', 'bmp'}
        """
        self.fore_image.save(filename, file_type=file_type)

    def save(self, filename, file_type='png'):
        """
        save the map image to the specified filename

        This copies the foreground image on top of the
        background image and saves the whole thing.

        :param filename: full path of file to be saved to
        :type filename: string

        :param file_type: type of file to save
        :type file_type: one of the following:
                         {'png', 'gif', 'jpeg', 'bmp'}
        """
        # copy the pallette from the foreground image
        # print dir(self.fore_image)
        # print self.fore_image.colors
        # fixme: is this being used???
        assert False

        self.fore_image.copy(self.back_image,
                             (0, 0), (0, 0),
                             self.back_image.size)


class GridLines(object):
    """
    class to hold logic for determining where the gridlines should be
    for the graticule
    """
    DEGREE = np.float64(1.0)
    MINUTE = DEGREE / 60.0
    SECOND = MINUTE / 60.0

    DMS_STEPS = (SECOND * 15,
                 SECOND * 30,
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
                 DEGREE * 40)
    DMS_COUNT = len(DMS_STEPS)

    DEGREE = np.float64(1.0)
    TENTH = DEGREE / 10.0
    HUNDREDTH = DEGREE / 100.0
    THOUSANDTH = DEGREE / 1000.0

    DEG_STEPS = (THOUSANDTH,
                 THOUSANDTH * 2.5,
                 THOUSANDTH * 5,
                 HUNDREDTH,
                 HUNDREDTH * 2.5,
                 HUNDREDTH * 5,
                 TENTH,
                 TENTH * 2.5,
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
                 DEGREE * 40)
    DEG_COUNT = len(DEG_STEPS)

    def __init__(self, viewport=None, projection=None,
                 max_lines=10, DegMinSec=False):
        """
        Creates a GridLines instance that does the logic for and describes
        the current graticule

        :param viewport: bounding box of the viewport in question.
        :type viewport: tuple of lon/lat

        :param max_lines: How many lines to be displayed on the longest
                          dimension of the viewport. Graticule will scale up
                          or down only when the number of lines in the viewport
                          falls outside the range.

        :type max_lines: tuple of integers, (max, min)

        :param DegMinSec: Whether to scale by Degrees/Minute/Seconds,
                          or decimal lon/lat

        :type bool
        """
        if viewport is None:
            raise ValueError('Viewport needs to be provided to generate '
                             'grid lines')
        self.viewport = viewport

        if projection is None:
            raise ValueError('Projection needs to be provided to generate '
                             'grid lines')
        self.projection = projection

        if DegMinSec:
            self.STEPS = self.DMS_STEPS
            self.STEP_COUNT = self.DMS_COUNT
        else:
            self.STEPS = self.DEG_STEPS
            self.STEP_COUNT = self.DEG_COUNT

        self.DMS = DegMinSec

        # need to just use refresh_scale for this...
        self.max_lines = max_lines
        self.refresh_scale()

    def get_step_size(self, reference_size):
        """
        Chooses the interval size for the graticule, based on where the
        reference size fits into the STEPS table.

        :param reference_size: the approximate size you want in decimal degrees
        """
        return self.STEPS[min(bisect.bisect(self.STEPS, abs(reference_size)),
                              self.STEP_COUNT - 1,)
                          ]

    def get_lines(self):
        """
        Computes, builds, and returns a list of lines that when drawn,
        creates the graticule.
        The list consists of self.lon_lines vertical lines,
        followed by self.lat_lines horizontal lines.
        """
        if self.max_lines == 0:
            return []

        (minlon, minlat) = self.projection.image_box[0]

        # create array of lines
        top = ((self.lat_lines + 4) * self.current_interval)
        right = ((self.lon_lines + 4) * self.current_interval)

        vertical_lines = np.array([((x * self.current_interval, 0),
                                    (x * self.current_interval, top))
                                   for x in range(0, self.lon_lines + 4)])
        horizontal_lines = np.array([((0, y * self.current_interval),
                                      (right, y * self.current_interval))
                                     for y in range(0, self.lat_lines + 4)])

        # shift lines into position
        delta = ((minlon // self.current_interval - 1) * self.current_interval,
                 (minlat // self.current_interval - 1) * self.current_interval)
        vertical_lines += delta
        horizontal_lines += delta

        return np.vstack((vertical_lines, horizontal_lines))

    def refresh_scale(self):
        """
        Recomputes the interval and number of lines in each dimension.
        This should be called whenever the viewport changes.
        """
        if self.max_lines == 0:
            return
        img_width = float(self.projection.image_size[0])
        img_height = float(self.projection.image_size[1])

        ratio = img_width / img_height
        self.ref_dim = 'w' if img_width >= img_height else 'h'

        width = (self.projection.image_box[1][0] -
                 self.projection.image_box[0][0])
        height = (self.projection.image_box[1][1] -
                  self.projection.image_box[0][1])

        self.ref_len = width if self.ref_dim == 'w' else height
        self.current_interval = self.get_step_size(self.ref_len /
                                                   self.max_lines)

        self.lon_lines = self.max_lines if self.ref_dim == 'w' else None
        self.lat_lines = self.max_lines if self.ref_dim == 'h' else None

        if self.lon_lines is None:
            self.lon_lines = int(round(self.lat_lines * ratio))

        if self.lat_lines is None:
            self.lat_lines = int(round(self.lon_lines / ratio))

    def set_max_lines(self, max_lines=None):
        """
        Alters the number of lines drawn.

        :param max_lines: the maximum number of lines drawn.
                          (Note: this is NOT the number of lines on the screen
                          at any given time.  That is determined by
                          the computed interval and the size/location
                          of the viewport)

        :type max_lines: int

        """

        if max_lines is not None:
            self.max_lines = max_lines

        self.refresh_scale()

    def set_DMS(self, DMS=False):
        '''
        :param DMS: Boolean value that specifies if Degrees/Minutes/Seconds
                    tags are enabled.
        :type DMS: Bool
        '''
        self.DMS = DMS

        if self.DMS:
            self.STEPS = self.DMS_STEPS
            self.STEP_COUNT = self.DMS_COUNT
        else:
            self.STEPS = self.DEG_STEPS
            self.STEP_COUNT = self.DEG_COUNT

        self.refresh_scale()

    def get_tags(self):
        """
        Returns a list of tags for each line (in the same order the lines
        are returned) and the position where the tag should be printed.

        Line labels are anchored at the intersection between the line and the
        edge of the viewport. This may cause the longitude labels to disappear
        if the aspect ratio of the image and viewport are identical.
        """
        if self.max_lines == 0:
            return []

        tags = []
        for line in self.get_lines():
            value = 0
            if line[0][0] == line[1][0]:
                value = line[0][0]
                hemi = 'E' if value > 0 else 'W'
            else:
                value = line[0][1]
                hemi = 'N' if value > 0 else 'S'
            tag = ("{0:.2f}".format(value)
                   if not self.DMS
                   else uc.LatLongConverter.ToDegMinSec(value, ustring=False))

            if self.DMS:
                degrees = int(abs(tag[0]))
                minutes = int(tag[1])
                seconds = int(round(tag[2]))

                if seconds == 60:
                    minutes += 1
                    seconds = 0

                if seconds != 0:
                    tag = "%id%i'%i\"%c" % (degrees, minutes, seconds, hemi)
                elif minutes != 0:
                    tag = "%id%i'%c" % (degrees, minutes, hemi)
                else:
                    tag = "%id%c" % (degrees, hemi)

            top = self.projection.image_box[1][1]
            left = self.projection.image_box[0][0]
            anchor = ((value, top)
                      if hemi == 'E' or hemi == 'W'
                      else (left, value))

            tags.append((tag, anchor))

        return tags


class Viewport(object):
    """
    Viewport

    class that defines and manages attribues for a viewport onto a flat 2D map.
    All points and measurements are in lon/lat
    """

    def __init__(self, BB=None, center=None, width=None, height=None):
        """
        Init the viewport.

        Can initialize with center/width/height, and/or with bounding box.
        NOTE: Bounding box takes precedence over any previous parameters

        :param center: The point around which the viewport is centered
        :type center: a tuple containing an lon/lat coordinate

        :param width: Width of the viewport (lon)

        :param height: height of the viewport (lat)

        :param BB: Bounding box of the viewport (overrides previous parameters)
        :type BB:  a list of lon/lat tuples containing the lower left
                   and top right coordinates
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

        self._center = (self.BB[1][0] - self.width / 2.0,
                        self.BB[1][1] - self.height / 2.0)

    def recompute_BB(self):
        halfx = self.width / 2.0
        halfy = self.height / 2.0

        self._BB = ((self.center[0] - halfx, self.center[1] - halfy),
                    (self.center[0] + halfx, self.center[1] + halfy))

    def aspect_ratio(self):
        return self.width / self.height

    @property
    def BB(self):
        return self._BB

    @BB.setter
    def BB(self, BB):
        self._BB = BB if BB is not None else self._BB
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
