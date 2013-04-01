#!/usr/bin/env python

"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for the interactive core mapping window -- at least for
the web version

"""
import copy
import os

import numpy as np
import PIL.Image
import PIL.ImageDraw

from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections, serializable
from gnome import basic_types, GnomeId
import gnome

def make_map(bna_filename, png_filename, image_size = (500, 500)):
    """
    utility function to draw a PNG map from a BNA file
    
    param: bna_filename -- file name of BNA file to draw map from
    param: png_filename -- file name of PNG file to write out
    param: image_size=(500,500) -- size of image (width, height) tuple
    param: format='RGB' -- format of image. Options are: 'RGB', 'palette', 'B&W'
    
    """


    #print "Reading input BNA"
    polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

    canvas = MapCanvas(image_size)
    
    canvas.set_land(polygons)
    canvas.draw_background()
    
    canvas.save_background(png_filename, "PNG")

        
class MapCanvas(object):
    """
    A class to hold and generate a map for GNOME
    
    This will hold (or call) all the rendering code, etc.
    
    This version uses PIL for the rendering, but it could be adapted to use other rendering tools
        
    This version uses a paletted image
    
    Note: For now - this is not serializable. Change if required in the future
    """
    # a bunch of constants -- maybe they should be settable, but...
    colors_rgb = [('transparent', (122, 122, 122) ),
                  ('background',  (255, 255, 255) ),
                  ('lake',        (255, 255, 255) ),
                  ('land',        (255, 204, 153) ),
                  ('LE',          (  0,   0,   0) ),
                  ('uncert_LE',   (255,   0,   0) ),
                  ('map_bounds',  (175, 175, 175) ),
                  ]

    colors  = dict( [(i[1][0], i[0]) for i in enumerate(colors_rgb)] )
    palette = np.array( [i[1] for i in colors_rgb], dtype=np.uint8 ).reshape((-1,))

    def __init__(self, image_size, **kwargs):
        """
        create a new map image from scratch -- specifying the size:
        
        :param size: (width, height) tuple of the image size in pixels
        :param projection_class: gnome.utilities.projections class to use.
        :param mode='RGB': image mode to use -- format of image. Options are: 'RGB', 'palette', 'B&W'
        
        Optional parameters (kwargs)
        :param projection_class:
        :param id:
        :param viewport:
        """
        self.image_size = image_size
        self.back_image = PIL.Image.new('P', self.image_size, color=self.colors['background'])
        self.back_image.putpalette(self.palette)
        
        # optional arguments (kwargs)
        self.land_polygons = kwargs.pop('land_polygons', None)
        
        # should user be able to change map_BB and viewport?
        self.map_BB = kwargs.pop('map_BB', None)    
        
        if self.map_BB is None:
            if self.land_polygons is None:
                self.map_BB = ((-180,-90),(180, 90))
            else:
                self.map_BB = self.land_polygons.bounding_box
        
        projection_class= kwargs.pop('projection_class', projections.FlatEarthProjection)
        self.projection = projection_class(self.map_BB, self.image_size) # BB will be re-set
        
        self._viewport = kwargs.pop('viewport',None)
        
        if self._viewport is None:
            self.viewport = self.map_BB
        
        self._gnome_id = GnomeId(id=kwargs.pop('id',None))

    id = property( lambda self: self._gnome_id.id)

    @classmethod
    def empty_map(cls, size, bounding_box):
        """
        Alternative constructor for a map_canvas with no land
        
        :param size: the size of the image: (width, height) in pixels 
        
        :param bounding_box: the bounding box you want the map to cover, in teh form:
                             ( (min_lon, min_lat),
                               (max_lon, max_lat) )
        """
        mc = cls.__new__(cls)
        mc.image_size = size
        mc.back_image = PIL.Image.new('P', size, color=cls.colors['background'])
        mc.back_image.putpalette(mc.palette)
        mc.projection = projections.FlatEarthProjection(bounding_box, size)
        mc.map_BB = bounding_box 
        mc.land_polygons=None 

        return mc

    @property
    def viewport(self):
        """ returns the current value of viewport of map: what gets drawn and on what scale """
        return self._viewport
    
    @viewport.setter
    def viewport(self, viewport_BB):
        """
        Sets the viewport of the map: what gets drawn at what scale

        :param viewport_BB: the new viewport, as a BBox object, or in the form:
                            ( (min_long, min_lat),
                              (max_long, max_lat) )
        """
        self._viewport = viewport_BB
        self.projection.set_scale(viewport_BB, self.image_size)
    
    def set_land(self, polygons):
        """ todo: need to fix this - viewport is coupled here! """
        self.land_polygons = polygons
        self.map_BB = polygons.bounding_box # not sure if we want to do this
        #self.projection.set_scale(self.map_BB, self.image_size)
        self.viewport = self.map_BB
#===============================================================================
#    def set_land(self, polygons, BB=None):
#        """
#        sets the land polygons and optionally reset projection to fit
# 
#        :param polygons:  a geometry.polygons object, holding the land polygons
#        :param BB:  the bounding box (in lat, long) of the resulting image
#                    if BB == 'keep' the current scaling, etc is used
#                    if BB is None, the bounding box will be determined from
#                       the input polygons
#        """
#        self.land_polygons = polygons
#        
#        if self.map_BB is None:
#            """ set only if not previously set """
#            self.map_BB = polygons.bounding_box
#        if BB is None:
#            # find the bounding box from the polygons
#            self.projection.set_scale(self.map_BB, self.image_size)
#        elif BB != 'keep': # if it's keep, don't change the scale
#            self.projection.set_scale(self.land_BB, self.image_size)
#===============================================================================

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
        #self.back_image.save("empty_BW_image.png")
        if self.land_polygons: # is there any land to draw?
            # project the data:
            polygons = self.land_polygons.Copy()
            polygons.TransformData(self.projection.to_pixel_2D)
        
            drawer = PIL.ImageDraw.Draw(self.back_image)

            #fixme: should we make sure to draw the lakes after the land???
            for p in polygons:
                if p.metadata[1].strip().lower() == "map bounds":
                    #Draw the map bounds polygon
                    poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                    drawer.polygon(poly, outline=self.colors['map_bounds'])
                elif p.metadata[1].strip().lower() == "spillablearea":
                    # don't draw the spillable area polygon
                    continue
                elif p.metadata[2] == "2": #this is a lake
                    poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                    drawer.polygon(poly, fill=self.colors['lake'])
                else:
                    poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                    drawer.polygon(poly, fill=self.colors['land'])
        return None
    
    def create_foreground_image(self):
        self.fore_image_array = np.zeros((self.image_size[1],self.image_size[0]), np.uint8)
        self.fore_image = PIL.Image.fromarray(self.fore_image_array, mode='P')
        self.fore_image.putpalette(self.palette)

    def draw_elements(self, spill):
        """
        Draws the individual elements to a foreground image
        
        :param spill: a spill object to draw
        """
        ##fixme: add checks for the status flag (beached, etc)!
        if spill.num_elements > 0: # nothing to draw if no elements
            if spill.uncertain:
                color = self.colors['uncert_LE']
            else:
                color = self.colors['LE']
                
            positions = spill['positions']

            pixel_pos = self.projection.to_pixel(positions, asint=False)
            arr = self.fore_image_array

            # remove points that are off the map
            pixel_pos = pixel_pos[(pixel_pos[:,0] > 1) &
                                  (pixel_pos[:,1] > 1) &
                                  (pixel_pos[:,0] < (self.image_size[0]-2) ) &
                                  (pixel_pos[:,1] < (self.image_size[1]-2) ) ]
            # draw the four pixels for the LE
            #note: long-lat backwards for array (vs image)
            arr[(pixel_pos[:,1]-0.5).astype(np.int32), (pixel_pos[:,0]-0.5).astype(np.int32)] = color
            arr[(pixel_pos[:,1]-0.5).astype(np.int32), (pixel_pos[:,0]+0.5).astype(np.int32)] = color
            arr[(pixel_pos[:,1]+0.5).astype(np.int32), (pixel_pos[:,0]-0.5).astype(np.int32)] = color
            arr[(pixel_pos[:,1]+0.5).astype(np.int32), (pixel_pos[:,0]+0.5).astype(np.int32)] = color


    def save_background(self, filename, type_in="PNG"):
        self.back_image.save(filename, type_in)

    def save_foreground(self, filename, type_in="PNG"):
        self.fore_image.save(filename, transparency=self.colors['transparent'])
    
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
    land_color       = 1
    lake_color       = 0 # same as background -- i.e. water.

    # a bunch of constants -- maybe they should be settable, but...
    colors_BW = [ ('transparent', 0 ), # note:transparent not really supported
                  ('background',  0 ),
                  ('lake',        0 ),
                  ('land',        1 ),
                  ('LE',          255 ),
                  ('uncert_LE',   255 ),
                  ('map_bounds',  0 ),
                  ]

    colors  = dict( colors_BW )
    
    def __init__(self, size, projection_class=projections.FlatEarthProjection):
        """
        create a new B&W map image from scratch -- specifying the size:
        
        :param size: (width, height) tuple of the image size
        :param projection_class: gnome.utilities.projections class to use.
        
        """
        self.image_size = size
        ##note: type "L" because type "1" didn't seem to give the right numpy array
        self.back_image = PIL.Image.new('L', self.image_size, color=self.colors['background'])
        #self.back_image = PIL.Image.new('L', self.image_size, 1)
        self.projection = projection_class(((-180,-85),(180, 85)), self.image_size) # BB will be re-set
        self.map_BB = None

    def as_array(self):
        """
        returns a numpy array of the data in the image
        
        this version returns dtype: np.uint8

        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        return np.ascontiguousarray(np.asarray(self.back_image, dtype=np.uint8).T)        
    
    

#if __name__ == "__main__":
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
    
class MapCanvasFromBNA(MapCanvas, serializable.Serializable):
    """ 
    extends the MapCanvas class to initialize from BNA
    
    This class is serializable 
    """
    _update = ['viewport','map_BB']
    _create = ['image_size','filename','projection_type']   # not sure image_size should be updated
    _create.extend(_update)
    state = copy.deepcopy(serializable.Serializable.state)
    state.add( create=_create, update=_update)
    
    @classmethod
    def new_from_dict(cls, dict_):
        """
        change projection_type from string to correct type 
        """
        proj = eval(dict_.pop('projection_type'))
        return cls(projection_class=proj, **dict_)
    
    def __init__(self, image_size, filename, **kwargs):
        if not os.path.exists(filename):
            raise IOError("{0} does not exist. Enter a valid BNA file".format(filename))
        
        self.filename = filename
        polygons = haz_files.ReadBNA(filename, "PolygonSet")
        
        MapCanvas.__init__(self, image_size, land_polygons=polygons, **kwargs)

    def projection_type_to_dict(self):
        """ store projection class as a string for now since that is all that is required for persisting """
        return "{0}.{1}".format(self.projection.__module__, self.projection.__class__.__name__)
        