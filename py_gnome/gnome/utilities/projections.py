#!/usr/bin/env python

"""
Module to hold classes and supporting code for projections used in GNOME.

Only:

  * no projection
  * geo-projection (just scaling to pixels)
  * a simple "flat earth" projection for 

Also a bit of code for scaling lat-long to meters, etc.

Used by map_canvas code and map code.

NOTE: all coordinates are takes as (lon, lat, depth (even though depth is always ignored
"""

import sys
import numpy as np

class NoProjection(object):
    """
    This is do-nothing projection class -- returns what it gets.
    
    It optionally rounds down to integer (pixel) coordinates
    
    used for testing, primarily, and as a definition of the interface
    """
    def __init__(self, bounding_box=None, image_size=None):
        """
        create a new do-nothing projection
        """
        pass
        return None
    
    def set_scale(self, bounding_box, image_size):
        pass

    def to_pixel(self, coords, asint=False):
        """
        returns the same (lon, lat) coords, but as an np.array, if they aren't already
        
        :param coords: -- the coords to project (Nx3 numpy array or compatible sequence)
                          (lon, lat, depth)
        :param asint: -- flag to set whether to convert to a integer or not default
                         is to leave it as the same type it came in, so you can have fractional pixels
        """
        if asint:
            return np.asarray(coords, dtype=np.int).reshape((-1,3))[:,:2]
        else:
            return np.asarray(coords).reshape((-1,3))[:,:2]

    def to_pixel_2D(self, coords, asint=False):
        """
        same as to_pixel, but expects only (lon, lat) coords as input.

        :param coords: -- the coords to project (Nx2 numpy array or compatible sequence)
                          (lon, lat)
        :param asint: -- flag to set whether to convert to a integer or not default
                         is to leave it as the same type it came in, so you can have fractional pixels
        """
        if asint:
            return np.asarray(coords, dtype=np.int)
        else:
            return np.asarray(coords)


    def to_lonlat(self, coords):
        """
        returns the same coords, but as a np.array of float64 , if they aren't already
        """
        return np.asarray(coords, dtype=np.float64)


class GeoProjection(object):
    """
    This acts as the base class for a projection
    
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
        set (or reset) the scaling, etc. of the projection
        
        This should be called whenever the view, bounding box, or size of image is changed.
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

    def to_pixel(self, coords, asint=False):
        """
        
        converts input coordinates to pixel coords
        
        param: coords --  an array of coordinates:
          NX3: ( (long1, lat1, z1),
                 (long2, lat2, z2),
                 (long3, lat3, z3),
                 .....
                )
        (z is ignored, and there is no z in the returned array)
        
        returns:  the pixel coords as a similar Nx2 array of integer x,y coordinates
        (using the y = 0 at the top, and y increasing down) -- a
        
        NOTE: the values between the minimum of a pixel value to less than the
              max of a pixel range are in that pixel, so  a point exactly at 
              the minimum of the bounding box will be in the zeroth pixel, but 
              a point  exactly at the max of the bounding box will be considered
              outside the map
        """
        coords = np.asarray(coords).reshape((-1,3))
        # shift to center:
        coords = coords[:, :2] - self.center
        # scale to pixels:
        coords *= self.scale
        # shift to pixel coords
        coords += self.offset

        if asint:
            # NOTE: using "floor" as it rounds negative numbers towards -inf
            ##      simple casting rounds toward zero
            ## we may need the negative coords to work right for locations off the grid.
            ##  (used for the raster map code)
            np.floor(coords, coords)
            return np.array(coords, np.int32) ##Assure C-contiguous
        else:
            return coords

    def to_pixel_2D(self, coords, asint=False):
        """
        same as to_pixel, but expects only (lon, lat) coords as input.

        :param coords: -- the coords to project (Nx2 numpy array or compatible sequence)
                          (lon, lat)
        :param asint: -- flag to set whether to convert to a integer or not default
                         is to leave it as the same type it came in, so you can have fractional pixels
        """
        coords = np.asarray(coords).reshape((-1,2))

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
            np.floor(coords, coords)
            return np.array(coords, np.int32) ##assure C-contiguous
        else:
            return coords
    
    def to_lonlat(self, coords):
        ## note: untested!
        """
        converts pixel coords to long-lat coords
        
        param: coords  - an array of pixel coordinates (usually integer type)
           NX2: ( (long1, lat1),
                  (long2, lat2),
                  (long3, lat3),
                 .....
                )
         (as produced by to_pixel)
        
        NOTE: there is not depth in input -- pixels are always 2-d!
        
        Note that  to_lonlat( to_pixel (coords) ) != coords, due to rounding.
        If the input is integers, a 0.5 is added to "shift" the location to mid-pixel.
        returns:  the pixel coords as a similar Nx2 array of floating point x,y coordinates
        (using the y = 0 at the top, and y increasing down)
         """
        
        if np.issubdtype(coords.dtype, int):
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
    
    @staticmethod
    def meters_to_lonlat(meters, ref_positions):
        """
        Converts from delta meters to delta latitude-longitude, using the Flat-Earth projection.
        
        :param meters: NX3 numpy array of (dx, dy, dz) distances in meters (dz is passed through untouched)
        :param ref_positions: NX3, numpy array of reference positions in degrees (Only lat is used here)

        :returns delta_lon_lat: Nx3 numpy array of (delta-lon, delta-lat, delta-z) triples

        dlat = dy * 8.9992801e-06

        dlon = dy * 8.9992801e-06 * cos(ref_lat) 

        (based on previous GNOME value: and/or average radius of the earth of 6366706.989  m)

        """
        #make a copy -- don't change meters
        delta_lon_lat = np.array(meters, dtype=np.float64).reshape(-1, 3)
        # reference is possible for reference positions
        ref_positions = np.asarray(ref_positions, dtype=np.float64).reshape(-1, 3)

        delta_lon_lat[:,:2] *= 8.9992801e-06
        delta_lon_lat[:,0] /= np.cos(np.deg2rad(ref_positions[:,1]))
        return delta_lon_lat

    @staticmethod
    def lonlat_to_meters(lon_lat, ref_positions):
        """
        Converts from delta longitude-latitude to delta meters, using the
        Flat-Earth projection. This shold be a reversal of meters_to_latlon.
        
        This function mainly used for testing

        :param lon_lat: NX3 numpy array of (dlon, dlat, dz) distances in meters (dz is passed through untouched)
        :param ref_positions: NX3, numpy array of (lon,lat,z) reference positions in degrees (Only lat is used here)

        :returns delta_meters: Nx3 numpy array of (delta-x, delta-y, delta-z) triples

        NOTE: the input is in units of longitude and latitude, but they are relative -- no absolute -- so 0
              means zero distance, not on the equator

        dy = dlon / 8.9992801e-06

        dx = dlat / ( 8.9992801e-06 * cos(ref_lat) ) 

        (based on previous GNOME value: and/or average radius of the earth of 6366706.989  m)

        """
        #make a copy -- don't change input
        delta_meters = np.array(lon_lat, dtype=np.float64).reshape(-1, 3)
        # reference is possible for reference positions
        ref_positions = np.asarray(ref_positions, dtype=np.float64).reshape(-1, 3)

        delta_meters[:,:2] /= 8.9992801e-06
        delta_meters[:,0] *= np.cos(np.deg2rad(ref_positions[:,1]))
        return delta_meters

    @staticmethod
    def geodesic_sphere(lon, lat, distance, bearing):
        """
        Given a start point, initial bearing, and distance, returns the
        destination point along a (shortest distance) great circle arc --
        assuming a spherical earth. Similar to how GNOME does it.
        
        :param lon: longitude in decimal degrees.
        :param lat: latitude in decimal degrees.
        :param distance:  meters.
        :param bearing: in decimal degrees, measured clockwise from north.
        
        :returns longitude, latitude: in degrees.
        
        Code from Brian Zelenke
        
        NOTE: performance could be improved a lot here if need be (lots of data copies)
        
        """
        # EarthRadius = 6371010.0 # Value I"ve looked up
        EarthRadius = 6366706.989 # Matches the value used above -- GNOME value

        #Convert from degrees to radians.
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        bearing = np.deg2rad(bearing)
        
        #Convert linear distance to angular distance (in radians).
        distance = distance/EarthRadius
        
        latout = np.arcsin(np.sin(lat)*np.cos(distance)+np.cos(lat)*np.sin(distance)*np.cos(bearing))
        lonout = lon+np.arctan2(np.sin(bearing)*np.sin(distance)*np.cos(lat),np.cos(distance)-np.sin(lat)*np.sin(latout))
        
        #Convert from radians to degrees.
        lonout = np.rad2deg(lonout)
        latout = np.rad2deg(latout)
        
        return lonout, latout
    
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
        w = (bounding_box[1,0] - bounding_box[0,0]) * lon_scale
        
        if w/h > image_size[0] / image_size[1]:
            s = image_size[0] / w
            self.scale = (s*lon_scale, -s)
        else:
            s = image_size[1] / h
            self.scale = (s*lon_scale, -s)
        

