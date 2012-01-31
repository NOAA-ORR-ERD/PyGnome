#!/usr/bin/env python

"""
A simple script to draw a GNOME map from a BNA
"""

import sys
import numpy as np

from hazpy.file_tools import haz_files

## note -- these shouldnt be globals, really.
image_size = (1400, 1000)
background_color = (255, 255, 255)
lake_color       = (0, 255, 255)
land_color       = ((255, 204, 153))

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

def draw_map(polygons, image_size):
    """
    draws the map to a PIL image, and returns the image 
    """
    from PIL import Image, ImageDraw

    im = Image.new('RGB', image_size, color=background_color)
    draw = ImageDraw.Draw(im)

    for p in polygons:
        try:
            int(p.metadata[1])
            poly = np.round(p).astype(np.int32).reshape((-1,)).tolist()
            draw.polygon(poly, fill=land_color)
        except ValueError: # not a regular polygon
            pass
    return im
    

class simple_projection:
    """
    class to hold a particular projection to/from la-long t pixels in an image

    simple_projection(bounding_box, image_size)
    
    bounding_box: the boudnign box of the map:
        ( (min_long, min_lat),
          (max_lon,  max_lat) )
        (or a BoundingBox Object)
    
    image_size: the size of the map image -- (width, height)

    """

    def __init__(self, bounding_box, image_size):
        """
        simple_projection(bounding_box, image_size)
    
        bounding_box: the boudnign box of the map:
            ( (min_long, min_lat),
              (max_lon,  max_lat) )

            (or a BoundingBox Object)
    
         image_size: the size of the map image -- (width, height)

        """
        self.set_scale(bounding_box, image_size)
    
    def set_scale(self, bounding_box, image_size):
        """
        set the scaling, etc of the projection
        
        This should be called whenever the baoudnign box of the map,
        or the size of the image is changed
        """
        
        bounding_box = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bounding_box, axis=0)
        self.offset = np.array((image_size), dtype=np.float64) / 2
        
        lon_scale = np.cos(np.deg2rad(self.center[1]))
        # compute BB to fit image
        h = bounding_box[1,1] - bounding_box[0,1]
        # width scaled to longitude
        w = lon_scale * (bounding_box[1,0] - bounding_box[0,0])
        if w/h > image_size[0] / image_size[1]:
            s = image_size[0] / w
        else:
            s = image_size[1] / h
        #s *= 0.5
        self.scale = (lon_scale*s, -s)
        

    def ToPixel(self, coords):
        # shift to center:
        coords = coords - self.center
        # scale to pixels:
        coords *= self.scale
        # shift to pixel coords
        coords += self.offset
        return coords
    
    def ToLatLon(self, coords):
        ## note: untested!

        # shift to pixel center coords
        coords = coords - self.offset
        # scale to lat-lon
        coords /= self.scale
        # shift from center:
        coords = coords + self.center

        return coords

if __name__ == "__main__":
#    # a small sample for testing:
#    bb = np.array(((-30, 45), (-20, 55)), dtype=np.float64)
#    im = (100, 200)
#    proj = simple_projection(bounding_box=bb, image_size=im)
#    print proj.ToPixel((-20, 45))
#    print proj.ToLatLon(( 50., 100.))

    bna_filename = sys.argv[1]
    png_filename = bna_filename.rsplit(".")[0] + ".png"
    bna = make_map(bna_filename, png_filename)
    
