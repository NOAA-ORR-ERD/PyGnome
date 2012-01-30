#!/usr/bin/env python

"""
A simple script to draw a GNOME map from a BNA
"""

import sys
import numpy as np

from hazpy.file_tools import haz_files

import map_canvas


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
    

class Projection:
    """
    This is the base class for a projection
    
    This one doesn't really project, but does convert to pixel coords
    """
    def __init__():
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
        # shift to center:
        coords = coords - self.center
        # scale to pixels:
        coords *= self.scale
        # shift to pixel coords
        coords += self.offset
        return coords
    
    def to_lat_long(self, coords):
        ## note: untested!

        # shift to pixel center coords
        coords = coords - self.offset
        # scale to lat-lon
        coords /= self.scale
        # shift from center:
        coords = coords + self.center

        return coords

class FlatEarthProjection(Projection):
    """
    class to define a "flat earth" projection:
        longitude is scaled to the cos of the mid-latitude -- but that's it
        
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
        h = bounding_box[1,1] - bounding_box[0,1]
        # width scaled to longitude
        w = lon_scale * (bounding_box[1,0] - bounding_box[0,0])
        if w/h > image_size[0] / image_size[1]:
            s = image_size[0] / w
        else:
            s = image_size[1] / h
        #s *= 0.5
        self.scale = (lon_scale*s, -s)
        
        
if __name__ == "__main__":
#    # a small sample for testing:
#    bb = np.array(((-30, 45), (-20, 55)), dtype=np.float64)
#    im = (100, 200)
#    proj = simple_projection(bounding_box=bb, image_size=im)
#    print proj.ToPixel((-20, 45))
#    print proj.ToLatLon(( 50., 100.))

    ## default image size
    image_size = (1400, 1000)

    HELP = """ map_from_bna.py
    
    script to create a png of a GNOME base map from a ban file
    
    usage:
      map_from_bna.py bna_filename [width height]
    
    if width and height are not given, the default: %s will be used
    """%(image_size,)

    num_args = len(sys.argv) - 1
    if num_args == 3:
        bna_filename = sys.argv[1]
        image_size = (int(sys.argv[2]), int(sys.argv[3]))
    elif num_args == 1:
        bna_filename = sys.argv[1]
    else:   
        print HELP
    
    png_filename = bna_filename.rsplit(".")[0] + ".png"

    print "Reading input BNA"
    polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

    print "number of input polys: %i"% len(polygons)
    print "total number of input points: %i "%polygons.total_num_points

    map = map_canvas.MapCanvas(size=image_size, projection=map_canvas.FlatEarthProjection)
    map.draw_land(polygons)

    map.save(png_filename)

    
