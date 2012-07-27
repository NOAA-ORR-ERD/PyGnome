#!/usr/bin/env python

"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for the interactive core mapping window -- at least for
the web version

"""

import sys
import numpy as np

from PIL import Image, ImageDraw

from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections


def make_map(bna_filename, png_filename, image_size = (500, 500), format='RGB'):
    """
    utility function to draw a PNG map from a BNA file
    
    param: bna_filename -- file name of BNA file to draw map from
    param: png_filename -- file name of PNG file to write out
    param: image_size=(500,500) -- size of image (width, height) tuple
    param: format='RGB' -- format of image. Options are: 'RGB', 'palette', 'B&W'
    
    """


    #print "Reading input BNA"
    polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

    if format == 'B&W':
        canvas = BW_MapCanvas(image_size)
    elif format == 'palette':
        canvas = Palette_MapCanvas(image_size)
    elif format == 'RGB':
        canvas = MapCanvas(image_size)
    else:
        raise ValueError("image format: %s not supported"%format)
    
    canvas.draw_land(polygons)
    
    canvas.save(png_filename, "PNG")

        
class MapCanvas:
    """
    A class to hold and generate a map for GNOME
    
    This will hold (or call) all the rendering code, etc.
    
    This version uses PIL for the rendering, but it could be adjusted to use other rendering tools
    
    fixme: this should be able to auto-shape the map to the aspect ratio of the map
    
    """
    # a bunch of constants -- maybe they should be settable, but...
    background_color = (255, 255, 255)
    #lake_color       = (0, 128, 255) # blue
    lake_color       = background_color
    land_color       = (255, 204, 153)

    def __init__(self, size, projection=projections.FlatEarthProjection, mode='RGB'):
        """
        create a new map image from scratch -- specifying the size:
        
        size: (width, height) tuple
        
        """
        self.image = Image.new(mode, size, color=self.background_color)
        self.projection = projection
    
    def draw_land(self, polygons, BB=None):
        """
        Draws the map to the internal image.
        
        Note that the land map is retained in the internal image.
        
        polygons is a geometry.polygons object, holding the land polygons
        BB is the bounding box (in lat, long) of the resulting image
           if BB is not provided, the bounding box will be determined from the land polygons
        
        """
        # find the bounding box:
        BB = polygons.bounding_box
        self.projection = self.projection(BB, self.image.size)
        # project the data:
        polygons.TransformData(self.projection.to_pixel)
        
        drawer = ImageDraw.Draw(self.image)

        for p in polygons:
            if p.metadata[1].strip().lower() == "map bounds":
                #Don't draw the map bounds polygon
                continue
            elif p.metadata[1].strip().lower() == "spillablearea":
                # don't draw the spillable area polygon
                continue
            elif p.metadata[2] == "2": #this is a lake
                poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                drawer.polygon(poly, fill=self.lake_color)
            else:
                poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                drawer.polygon(poly, fill=self.land_color)
        return None
    
    def draw_particles(self, spills, filename):
        img = self.image.copy()
        for spill in spills:
            rgbt = (0,0,0)
            if spill.uncertain:
                rgbt = (255,0,0)
            pra = spill.npra['p']
            for i in xrange(0, len(pra)):
                xy = self.to_pixel((pra[i]['p_long'], pra[i]['p_lat']))
                try:
                    img.putpixel(xy, rgbt)
                except: # fixme! what exception are we catching?
                    pass
        img.save(filename)
    
    def save(self, filename, type="PNG"):
        self.image.save(filename, type)
        
    def as_array(self):
        """
        returns a numpy array of the data in the image
        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        ## fixme: what data type and shape will this give us?
        return np.ascontiguousarray(np.asarray(self.image).T)

class Palette_MapCanvas(MapCanvas):
    """
    a version of the map canvas that uses a palleted image:
    256 colors only.
    
    """
    
    def __init__(self, size, projection=projections.FlatEarthProjection):
        """
        create a new map image from scratch -- specifying the size:
        
        size: (width, height) tuple
        
        """
        self.image = Image.new('P', size, color=0)
        drawer = ImageDraw.Draw(self.image) # couldn't find a better way to initilize the colors right.
        drawer.rectangle(((0,0), size), fill=self.background_color)
        
        self.projection = projection

    def as_array(self):
        """
        returns a numpy array of the data in the image
        
        this version returns dtype: np.uint8

        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        return np.ascontiguousarray(np.asarray(self.image, dtype=np.uint8).T)
    
class BW_MapCanvas(MapCanvas):
    """
    a version of the map canvas that draws Black and White images
    (Note -- hard to see -- water color is very, very dark grey!)
    used to generate the raster maps
    """
    background_color = 0
    land_color       = 1
    lake_color       = 0 # same as background -- i.e. water.
    
    def __init__(self, size, projection=projections.FlatEarthProjection):
        """
        create a new map image from scratch -- specifying the size:
        
        size: (width, height) tuple
        
        """
        self.image = Image.new('L', size, color=self.background_color)
        self.projection = projection

    def as_array(self):
        """
        returns a numpy array of the data in the image
        
        this version returns dtype: np.uint8

        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        return np.ascontiguousarray(np.asarray(self.image, dtype=np.uint8).T)

    

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
    
