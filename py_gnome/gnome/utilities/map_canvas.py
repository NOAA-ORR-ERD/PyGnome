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

        
class MapCanvas(object):
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
    element_color    = (0, 0, 0)
    uncert_element_color = (255, 0, 0)

    def __init__(self, size, projection=projections.FlatEarthProjection, mode='RGB'):
        """
        create a new map image from scratch -- specifying the size:
        
        size: (width, height) tuple
        
        """
        print "init-ing an RGB mapcanvas"
        self.image = Image.new(mode, size, color=self.background_color)
        self.projection = projection(((-180,-85),(180, 85)), size) # BB will be re-set
    
    def draw_land(self, polygons=None, BB=None):
        """
        Draws the map to the internal image.
        
        Note that the land map is retained in the internal image.
        
        polygons is a geometry.polygons object, holding the land polygons
        BB is the bounding box (in lat, long) of the resulting image
           if BB == 'keep' the current scaling, etc is used
           if BB is None, the bounding box will be determined from the input polygons
        
        """
        if BB is None:
            # find the bounding box from the polygons
            BB = polygons.bounding_box
            print "setting the scale from the polygons"
            self.projection.set_scale(BB, self.image.size)
        elif BB <> 'keep': # if it's keep, don't change the scale
            self.projection.set_scale(BB, self.image.size)
        # project the data:
        polygons.TransformData(self.projection.to_pixel_2D)
        
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
    
    def draw_elements(self, spills, filename):
        """
        Draws the individual elements
        
        :param spills: a list of spill object to draw
        :param filename: the filename to output the final image to
        
        Makes a copy of the base map, then draw the LEs on top of it
        """
        # this way can use a pixel access object -- but numpy probably better anyway
        img = self.image.copy().load() # gives a pixel access object
        img = self.image.copy()
        drawer = ImageDraw.Draw(img)
        
        for spill in spills:
            if spill.is_uncertain:
                color = self.uncert_element_color
            else:
                color = self.element_color
            positions = spill['positions']
            pixel_pos = self.projection.to_pixel(positions, asint=True)
            for point in pixel_pos:
                drawer.point(point, fill=color)
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
    a version of the map canvas that uses a paletted image:
    256 colors only.
    """
    
    def __init__(self, size, projection=projections.FlatEarthProjection):
        """
        create a new map image from scratch -- specifying the size:
        
        :param size: (width, height) tuple
        
        """
        self.image = Image.new('P', size, color=0)
        drawer = ImageDraw.Draw(self.image) # couldn't find a better way to initilize the colors right.
        drawer.rectangle(((0,0), size), fill=self.background_color)
        
        self.projection = projection(((-180,-85),(180, 85)), size) # BB will be re-set

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
        self.projection = projection(((-180,-85),(180, 85)), size) # BB will be re-set

    def as_array(self):
        """
        returns a numpy array of the data in the image
        
        this version returns dtype: np.uint8

        """
        # makes sure the you get a c-contiguous array with width-height right
        #   (PIL uses the reverse convention)
        return np.ascontiguousarray(np.asarray(self.image, dtype=np.uint8).T)


class GNOME_renderer(Palette_MapCanvas):
    """
    a map rendrer specifically for the WebGNOME use
    
    this may not be the best stucture, but I wanted to not break the existing API...
    """
    
    def __init__(self, map_file=None, projection=projections.FlatEarthProjection):
        """
        create a GNOME renderer
        
        if a file name is given, the base map is drawn from that BNA file
        
        """
        
        if map_file is not None:
            self.land_polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
        
        self.image = Image.new('P', size, color=0)
        drawer = ImageDraw.Draw(self.image) # couldn't find a better way to initilize the colors right.
        drawer.rectangle(((0,0), size), fill=self.background_color)
        
        self.projection = projection

        
        self.image = Image.new(mode, size, color=self.background_color)
        self.projection = projection

    def reset_view(self, size, viewport):
        """
        sets the image view -- how big the image is, and what is in view
        
        """
        pass
        
    
    

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
    
