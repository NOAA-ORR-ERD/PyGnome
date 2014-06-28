#!/usr/bin/env python
"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for the interactive core mapping window -- at least for
the web version.

This should have the basic drawing stuff. Specific rendering, like
dealing with spill_containers, etc, should be in the rendere subclass.
"""
import numpy
np = numpy

from PIL import Image, ImageDraw

from gnome.basic_types import oil_status

from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections


def make_map(bna_filename, png_filename, image_size=(500, 500)):
    """
    utility function to draw a PNG map from a BNA file

    param: bna_filename -- file name of BNA file to draw map from
    param: png_filename -- file name of PNG file to write out
    param: image_size=(500,500) -- size of image (width, height) tuple
    param: format='RGB' -- format of image.
                           Options are:
                           - 'RGB'
                           - 'palette'
                           - 'B&W'
    """
    # print "Reading input BNA"

    polygons = haz_files.ReadBNA(bna_filename, 'PolygonSet')

    canvas = MapCanvas(image_size, land_polygons=polygons)

    canvas.draw_background()
    canvas.save_background(png_filename, 'PNG')


class MapCanvas(object):
    """
    A class to hold and generate a map for GNOME

    This will hold (or call) all the rendering code, etc.

    This version uses PIL for the rendering, but it could be adapted to use
    other rendering tools

    This version uses a paletted image

    Note: For now - this is not serializable. Change if required in the future
    """

    # a bunch of constants -- maybe they should be settable, but...

    colors_rgb = [('transparent', (122, 122, 122)),
                  ('background', (255, 255, 255)),
                  ('lake', (255, 255, 255)),
                  ('land', (255, 204, 153)),
                  ('LE', (0, 0, 0)),
                  ('uncert_LE', (255, 0, 0)),
                  ('map_bounds', (175, 175, 175)),
                  ]

    colors = dict([(i[1][0], i[0]) for i in enumerate(colors_rgb)])
    palette = np.array([i[1] for i in colors_rgb],
                       dtype=np.uint8).reshape((-1, ))

    def __init__(self, image_size, land_polygons=None,
                 **kwargs):
        """
        create a new map image from scratch -- specifying the size:
        Only the "Palette" image mode to used for drawing image.

        :param image_size: (width, height) tuple of the image size in pixels

        Optional parameter:
        :param land_polygons: a PolygonSet
                             (gnome.utilities.geometry.polygons.PolygonSet)
                             used to define the map.
                             If this is None, MapCanvas has no land.
                             Set during object creation.

        Optional parameters (kwargs)
        :param projection_class: gnome.utilities.projections class to use.
                                 Default is
                                 gnome.utilities.projections.FlatEarthProjection
        :param map_BB: map bounding box.
                       Default is to use land_polygons.bounding_box.
                       If land_polygons is None, then this is the whole world,
                       defined by ((-180,-90),(180, 90))
        ## :param viewport: viewport of map -- what gets drawn and on what
                            scale. Default is to set viewport = map_BB
        :param image_mode: Image mode ('P' for palette or 'L' for
                           Black and White image)
                           BW_MapCanvas inherits from MapCanvas and sets the
                           mode to 'L'
                           Default image_mode is 'P'.
        """
        self.image_size = image_size
        self.image_mode = kwargs.pop('image_mode', 'P')

        self.back_image = None

        # optional arguments (kwargs)

        self._land_polygons = land_polygons
        self.map_BB = kwargs.pop('map_BB', None)

        if self.map_BB is None:
            if self.land_polygons is None:
                self.map_BB = ((-180, -90), (180, 90))
            else:
                self.map_BB = self.land_polygons.bounding_box

        projection_class = kwargs.pop('projection_class',
                                      projections.FlatEarthProjection)
        # BB will be re-set
        self.projection = projection_class(self.map_BB, self.image_size)

        # assorted status flags:

        self.draw_map_bounds = True

        # self._viewport = kwargs.pop('viewport',None)

        # if self._viewport is None:
        #     self.viewport = self.map_BB

# USE DEFAULT CONSTRUCTOR FOR CREATING EMPTY_MAP
# =============================================================================
#    @classmethod
#    def empty_map(cls, image_size, bounding_box):
#        """
#        Alternative constructor for a map_canvas with no land
#
#        :param image_size: the size of the image: (width, height) in pixels
#
#        :param bounding_box: the bounding box you want the map to cover,
#                             in teh form:
#                             ((min_lon, min_lat),
#                              (max_lon, max_lat))
#        """
#        mc = cls.__new__(cls)
#        mc.image_size = image_size
#        mc.back_image = PIL.Image.new('P', image_size,
#                                      color=cls.colors['background'])
#        mc.back_image.putpalette(mc.palette)
#        mc.projection = projections.FlatEarthProjection(bounding_box,
#                                                        image_size)
#        mc.map_BB = bounding_box
#        mc._land_polygons=None
#
#        return mc
# =============================================================================

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
        """
        #print "resetting viewport"
        self.projection.set_scale(viewport_BB, self.image_size)

    @property
    def land_polygons(self):
        return self._land_polygons

    def draw_background(self):
        """
        Draws the background image -- just land for now

        This should be called whenever the scale changes
        """
        self.draw_land()

    def draw_land(self):
        """
        Draws the land map to the internal background image.
        """
        back_image = Image.new(self.image_mode, self.image_size,
                                   color=self.colors['background'])
        back_image.putpalette(self.palette)

        # TODO: do we need to keep this around?

        self.back_image = back_image

        if self.land_polygons:  # is there any land to draw?

            # project the data:
            polygons = self.land_polygons.Copy()
            polygons.TransformData(self.projection.to_pixel_2D)

            drawer = ImageDraw.Draw(back_image)

            # TODO: should we make sure to draw the lakes after the land???

            for p in polygons:
                if p.metadata[1].strip().lower() == 'map bounds':
                    if self.draw_map_bounds:

                        # Draw the map bounds polygon
                        poly = (np.round(p)
                                .astype(np.int32)
                                .reshape((-1,))
                                .tolist())
                        drawer.polygon(poly, outline=self.colors['map_bounds'])
                elif p.metadata[1].strip().lower() == 'spillablearea':
                    # don't draw the spillable area polygon
                    continue
                elif p.metadata[2] == '2':
                    # this is a lake
                    poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                    drawer.polygon(poly, fill=self.colors['lake'])
                else:
                    poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                    drawer.polygon(poly, fill=self.colors['land'])
        return None

    def create_foreground_image(self):
        self.fore_image_array = np.zeros((self.image_size[1],
                                          self.image_size[0]), np.uint8)
        self.fore_image = Image.fromarray(self.fore_image_array, mode='P')
        self.fore_image.putpalette(self.palette)

    def draw_elements(self, sc):
        """
        Draws the individual elements to a foreground image

        :param sc: a SpillContainer object to draw
        """
        # TODO: add checks for the status flag (beached, etc)!
        if sc.num_released > 0:  # nothing to draw if no elements
            if sc.uncertain:
                color = self.colors['uncert_LE']
            else:
                color = self.colors['LE']

            positions = sc['positions']

            pixel_pos = self.projection.to_pixel(positions, asint=False)
            arr = self.fore_image_array

            # remove points that are off the view port
            on_map = ((pixel_pos[:, 0] > 1) &
                      (pixel_pos[:, 1] > 1) &
                      (pixel_pos[:, 0] < self.image_size[0] - 2) &
                      (pixel_pos[:, 1] < self.image_size[1] - 2))
            pixel_pos = pixel_pos[on_map]

            # which ones are on land?
            on_land = sc['status_codes'][on_map] == oil_status.on_land

            # draw the five "X" pixels for the on_land elements
            arr[pixel_pos[on_land, 1].astype(np.int32),
                pixel_pos[on_land, 0].astype(np.int32)] = color
            arr[(pixel_pos[on_land, 1] - 1).astype(np.int32),
                (pixel_pos[on_land, 0] - 1).astype(np.int32)] = color
            arr[(pixel_pos[on_land, 1] - 1).astype(np.int32),
                (pixel_pos[on_land, 0] + 1).astype(np.int32)] = color
            arr[(pixel_pos[on_land, 1] + 1).astype(np.int32),
                (pixel_pos[on_land, 0] - 1).astype(np.int32)] = color
            arr[(pixel_pos[on_land, 1] + 1).astype(np.int32),
                (pixel_pos[on_land, 0] + 1).astype(np.int32)] = color

            # draw the four pixels for the elements not on land and
            # not off the map
            off_map = sc['status_codes'][on_map] == oil_status.off_maps
            not_on_land = np.logical_and(~on_land, ~off_map)

            # note: long-lat backwards for array (vs image)
            arr[(pixel_pos[not_on_land, 1] - 0.5).astype(np.int32),
                (pixel_pos[not_on_land, 0] - 0.5).astype(np.int32)] = color
            arr[(pixel_pos[not_on_land, 1] - 0.5).astype(np.int32),
                (pixel_pos[not_on_land, 0] + 0.5).astype(np.int32)] = color
            arr[(pixel_pos[not_on_land, 1] + 0.5).astype(np.int32),
                (pixel_pos[not_on_land, 0] - 0.5).astype(np.int32)] = color
            arr[(pixel_pos[not_on_land, 1] + 0.5).astype(np.int32),
                (pixel_pos[not_on_land, 0] + 0.5).astype(np.int32)] = color

    def save_background(self, filename, type_in='PNG'):
        print 'saving:', filename
        self.back_image.save(filename, type_in)

    def save_foreground(self, filename, type_in='PNG'):
        self.fore_image.save(filename,
                             transparency=self.colors['transparent'])


    # Not sure this is required yet
#    def projection_pickle_to_dict(self):
#        """ returns a pickled projection object """
#        return pickle.dumps(self.projection)
#
#    def land_polygons_to_dict(self):
#        """ returns a pickled land_polygons object """
#        return pickle.dumps(self.land_polygons)

class BW_MapCanvas(MapCanvas):
    """
    a version of the map canvas that draws Black and White images
    (Note -- hard to see -- water color is very, very dark grey!)
    used to generate the raster maps
    """
    background_color = 0
    land_color = 1
    lake_color = 0  # same as background -- i.e. water.

    # a bunch of constants -- maybe they should be settable, but...
    # note:transparent not really supported
    colors_BW = [('transparent', 0),
                 ('background', 0),
                 ('lake', 0),
                 ('land', 1),
                 ('LE', 255),
                 ('uncert_LE', 255),
                 ('map_bounds', 0),
                 ]
    colors = dict(colors_BW)

    def __init__(self, image_size, land_polygons=None,
                 projection_class=projections.FlatEarthProjection):
        """
        create a new B&W map image from scratch -- specifying the size:

        :param image_size: (width, height) tuple of the image size
        :param land_polygons: a PolygonSet
                              (gnome.utilities.geometry.polygons.PolygonSet)
                              used to define the map.
                              If this is None, MapCanvas has no land.
                              This can be read in from a BNA file.
        :param projection_class: gnome.utilities.projections class to use.

        See MapCanvas documentation for remaining valid kwargs.
        It sets the image_mode = 'L' when calling MapCanvas.__init__
        """

        # =====================================================================
        # self.image_size = image_size
        # ##note: type "L" because type "1" didn't seem to give the right
        #         numpy array
        # self.back_image = PIL.Image.new('L', self.image_size,
        #                                 color=self.colors['background'])
        # #self.back_image = PIL.Image.new('L', self.image_size, 1)
        # # BB will be re-set
        # self.projection = projection_class(((-180,-85),(180, 85)),
        #                                    self.image_size)
        # self.map_BB = None
        # =====================================================================

        MapCanvas.__init__(self, image_size,
                           land_polygons=land_polygons,
                           projection_class=projections.FlatEarthProjection,
                           image_mode='L')

    def as_array(self):
        """
        returns a numpy array of the data in the background image

        this version returns dtype: np.uint8
        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        return np.ascontiguousarray(np.asarray(self.back_image,
                                               dtype=np.uint8).T)


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
