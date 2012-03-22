#!/usr/bin/env python

"""
Module to hold classes and suporting code for the map canvas for GNOME:

The drawing code for the interactive core mapping window -- at least for
the web version

"""

import sys
import numpy as np

from PIL import Image, ImageDraw

## note -- these shouldn't be globals, really.

def make_map(bna_filename, png_filename):
    print "Reading input BNA"
    polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

    print "number of input polys: %i"% len(polygons)
    print "total number of input points: %i "%polygons.total_num_points

    # find the bounding box:
    BB = polygons.bounding_box
    proj = simple_projection(BB, image_size)
    # project the data:
    polygons.TransformData(proj.ToPixel)
    
    im = draw_map(polygons, image_size)
    
    im.save(png_filename, "PNG")

class Projection:
    """
    This is the base class for a projection
    
    This one doesn't really project, but does convert to pixel coords
    i.e. "geo-coorddinaes"
    """
    def __init__(self, bounding_box, image_size):
        """
        create a new projection

        Projection(bounding_box, image_size)
    
        bounding_box: the bounding box of the map:
           ( (min_long, min_lat),
             (max_lon,  max_lat) )
        
        (or a BoundingBox Object)
        
        image_size: the size of the map image -- (width, height)
        """
        self.set_scale(bounding_box, image_size)
        
        return None
    
    def set_scale(self, bounding_box, image_size):
        """
        set (or reset) the scaling, etc of the projection
        
        This should be called whenever the bounding box of the map,
        or the size of the image is changed
        """
        
        bounding_box = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bounding_box, axis=0)
        self.offset = np.array((image_size), dtype=np.float64) / 2
        
        # compute BB to fit image
        h = bounding_box[1,1] - bounding_box[0,1]
        # width scaled to longitude
        w = (bounding_box[1,0] - bounding_box[0,0])
        if w/h > image_size[0] / image_size[1]:
            s = image_size[0] / w
        else:
            s = image_size[1] / h
        self.scale = (s, -s)

    def to_pixel(self, coords):
        """
        takes an array of coordinates:
          NX2: ( (long1, lat1),
                 (long2, lat2),
                 (long3, lat3),
                 .....
                )
        
        and returns the pixel coords as a similar Nx2 array of x,y coordinates
        (using the y = 0 at the top, and y increasing down)
        """
        # b = a.view(shape=(10,2),dtype='<f4')
        # shift to center:
        coords = coords - self.center
        # scale to pixels:
        coords *= self.scale
        # shift to pixel coords
        coords += self.offset
        
        return np.round(coords).astype(np.int)
    
    def to_lat_long(self, coords):
        ## note: untested!

        # shift to pixel center coords
        coords = coords - self.offset
        # scale to lat-lon
        coords /= self.scale
        # shift from center:
        coords += self.center
        return coords

class FlatEarthProjection(Projection):
    """
    class to define a "flat earth" projection:
        longitude is scaled to the cos of the mid-latitude -- but that's it.
        
        not conforming to eaual area, distance, bearing, or any other nifty
        map properties -- but easy to compute
        
    """
    
    def set_scale(self, bounding_box, image_size):
        """
        set the scaling, etc of the projection
        
        This should be called whenever the boudnign box of the map,
        or the size of the image is changed
        """
        
        bounding_box = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bounding_box, axis=0)
        self.offset = np.array((image_size), dtype=np.float64) / 2
        
        lon_scale = np.cos(np.deg2rad(self.center[1]))
        # compute BB to fit image
        h = bounding_box[1,1] -	 bounding_box[0,1]
        # width scaled to longitude
        w = lon_scale * (bounding_box[1,0] - bounding_box[0,0])
        if w/h > image_size[0] / image_size[1]:
            s = image_size[0] / w
        else:
            s = image_size[1] / h
        #s *= 0.5
        self.scale = (lon_scale*s, -s)
        
        
class MapCanvas:
    """
    A class to hold and generate a map for GNOME
    
    This will hold (or call) all the rendering code, etc.
    
    This version uses PIL for the rendering, but it could be adjusted to use other rendering tools
    
    """
    # a bunch of constants -- maybe they should be settable, but...
    background_color = (255, 255, 255)
    lake_color       = (0, 255, 255)
    land_color       = ((255, 204, 153))

    def __init__(self, size, projection=FlatEarthProjection, mode='RGB'):
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
            try:
                #int(p.metadata[1])
                temp_str = p.metadata[1].strip().lower()
                if temp_str == "map bounds" or temp_str == "spillablearea":
                    continue
                poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
                drawer.polygon(poly, fill=self.land_color)
                #print "done drawing"
                
            except ValueError: # not a regular polygon
                print 'exception!'
                
        print 'done drawing' + self._type()
        return None
    
    def draw_particles(self, spills, filename):
        img = self.image.copy()
        for spill in spills:
            pra = spill.npra['p']
            for i in xrange(0, len(pra)):
            	xy = self.to_pixel((pra[i]['p_long'], pra[i]['p_lat']))
                try:
            	    img.putpixel(xy, 1)
            	except:
				    pass
        img.save(filename)
    
    def save(self, filename, type="PNG"):
        self.image.save(filename, type)
    

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
    
