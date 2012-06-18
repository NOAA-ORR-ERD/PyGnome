#!/usr/bin/env python

"""
Module to hold classes and suporting code for projections used in GNOME.

Only a simple "flat earth projection for now.

Used by map_canvas code and map code.

The drawing code for the interactive core mapping window -- at least for
the web version

"""

import sys
import numpy as np

class NoProjection:
    """
    This is do-nothing projeciton class -- returns what it gets.
    
    It rounds down to integer (pixel) coordinates
    
    used for testing, primarily
    """
    def __init__(self, bounding_box=None, image_size=None):
        """
        create a new projection do-nothing projection
        """
        pass
        return None
    
    def set_scale(self, bounding_box, image_size):
        pass

    def to_pixel(self, coords, asint=True):
        """
        returns the same coords, but as an np.array , if they aren't already
        
        param: coords -- the coords to project (Nx2 numpy array or compatible sequence)
        param: asint -- flag to set whether to convert to a integer or not
        """
        if asint:
            return np.asarray(coords, dtype=np.int)
        else:
            return np.asarray(coords)

    def to_lat_long(self, coords):
        """
        returns the same coords, but as an np.array , if they aren't already
        """
        return np.AsArray(coords, dtype=np.float64)


class GeoProjection:
    """
    This acxts as the base class for a projection
    
    This one doesn't really project, but does convert to pixel coords
    i.e. "geo-coordinates"
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

    def to_pixel(self, coords, asint=True):
        """
        
        converts input coordinates to pixel coords
        
        param: coords --  an array of coordinates:
          NX2: ( (long1, lat1),
                 (long2, lat2),
                 (long3, lat3),
                 .....
                )
        
        returns:  the pixel coords as a similar Nx2 array of integer x,y coordinates
        (using the y = 0 at the top, and y increasing down) -- a
        
        NOTE: the values between the minimum of a pixel value to less than the
              max of a pixel range are in that pixel, so  a point exactly at 
              the minimum of the bounding box will be in the zeroth pixel, but 
              a point  exactly at the max of the bounding box will be considered
              outside the map
        """
        # b = a.view(shape=(10,2),dtype='<f4')
        # shift to center:
        coords = coords - self.center
        # scale to pixels:
        coords *= self.scale
        # shift to pixel coords
        coords += self.offset

        if asint:
            # NOTE: using "floor" as it rounds negative numbers towards -inf
            ##      simple casting rounds toward zero
            ## we may need the negative coords to work right for locations off the grid.
            ##  (used for the raster map code)
            return np.floor(coords, coords).astype(np.int)
        else:
            return coords

    
    def to_lat_long(self, coords):
        ## note: untested!
        """
        converts pixel coords to lat-long coords
        
        param: coords  - an array of pixel coordinates (usually integer type)
           NX2: ( (long1, lat1),
                  (long2, lat2),
                  (long3, lat3),
                 .....
                )
         (as produced by to_pixel)
        
        Note that  to_lat_long( to_pixel (coords) ) != coords, due to raounding.
        If the input is integers, a 0.5 is added to "shift" the location to mid-pixel.
        returns:  the pixel coords as a similar Nx2 array of floating point x,y coordinates
        (using the y = 0 at the top, and y increasing down)
         """
        
        if np.issubdtype(coords.dtype, int):
            print "it's integer data"
            # convert to float64:
            coords = coords.astype(np.float64)
            # add 0.5 to shift to center of pixel
            coords += 0.5
        # shift to pixel center coords
        coords -=  self.offset
        # scale to lat-lon
        coords /= self.scale
        # shift from center:
        coords += self.center
        
        return coords

class FlatEarthProjection(GeoProjection):
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
        self.scale = (lon_scale*s, -s)
        

