 #!/usr/bin/env python

"""
 An implementation of the GNOME land-water map.
 
 This is a port of the C++ raster map approach
 
"""
 
import sys
import os
import numpy as np
import random
from PIL import Image, ImageDraw

from gnome.utilities import map_canvas
from hazpy.file_tools import haz_files
from hazpy.geometry import polygons

class gnome_map(map_canvas.MapCanvas):
    
    """basic color bitmap."""
        
    def __init__(self, image_size, bna_filename, color_mode='RGB'):
        map_canvas.MapCanvas.__init__(self, image_size, projection=map_canvas.FlatEarthProjection, mode=color_mode)
        self.polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
        self.filename = bna_filename
        self.draw_land(self.polygons)

    def __del__(self):
        pass        

    def to_pixel(self, coord):
        coord = tuple(self.projection.to_pixel(np.array((coord[0], coord[1]))))
        coord = (int(coord[0]), int(coord[1]))
        return coord
        
    def to_pixel_array(self, coords):
        coords['p_long'] -= self.projection.center[0]
        coords['p_lat'] -= self.projection.center[1]
        coords['p_long'] *= self.projection.scale[0]
        coords['p_lat'] *= self.projection.scale[1]
        coords['p_long'] += self.projection.offset[0]
        coords['p_lat'] += self.projection.offset[1]
        coords['p_long'] = np.round(coords['p_long']).astype(np.int)
        coords['p_lat'] = np.round(coords['p_lat']).astype(np.int)
    
    
    def _type(self):
        return ' color bitmap'
        
class lw_map(gnome_map):

    """land-water bitmap."""
        
    background_color = 0
    lake_color = 0
    land_color = 1
    
    def __init__(self, image_size, bna_filename, refloat_halflife, color_mode='1'):
        gnome_map.__init__(self, image_size, bna_filename, color_mode)
        self.bounding_box = self.polygons.bounding_box
        self.refloat_halflife = refloat_halflife
        self.spills = []
    
    def __del__(self):
        pass

    def _type(self):
        return ' land-water bitmap'
        
    def on_map(self, pixel_coord):
        bounding_box = self.bounding_box
        if pixel_coord[0] > bounding_box[0][0] and pixel_coord[0] < bounding_box[1][0] and pixel_coord[1] > bounding_box[0][1] and pixel_coord[1] < bounding_box[1][1]:
            return True
        return False

    def on_land(self, coord):
        return not self.in_water(coord)

    def on_land_pixel(self, coord):
        return not self.in_water_pixel(coord)
        
    def in_water(self, coord):
        coord = self.to_pixel(coord)
        if not self.on_map(coord):
            return False
        try:
            chrom = self.image.getpixel(coord)
            if not chrom:
                return True
            else:
                return False
        except:
            print 'exception!',  sys.exc_info()[0]
            return False
            
    def in_water_pixel(self, coord):
        coord = coord.tolist()
        if not self.on_map(coord):
            return False
        try:
            chrom = self.image.getpixel(coord)
            if not chrom:
                return True
            else:
                return False
        except:
            print 'exception!',  sys.exc_info()[0]
            return False

    def allowable_spill_position(self, coord):
        return self.in_water(coord)
        
    def set_spill(self, coord, num_particles, release_time):
        if not self.allowable_spill_position(coord):
            print  "spill " + str(dict((('position', coord), ('num_particles', num_particles), ('release_time', release_time)))) + " ignored."
        else:
            self.spills += [(coord, num_particles, release_time)]

class LandWaterMap:
    """
    A GNOME map class -- determines where land and water are
    
    This is a base class that defines the interface -- with no implementation
    
    See subclasses for implementation
    """
    def __init__(self)):
        """
        This __init__ will be different depending on the implemenation
        """
        
        self.bounding_box = None
        self.refloat_halflife = None

    def on_map(self, coord):
        """
        returns a Boolean result:
        
        True if the point is on the map,
        False if not
        
        coord is a (long, lat) location
        
        """
        raise NotImplimentedError

    def on_land(self, coord):
        """
        returns a Boolean result:
        
        True if the point is on land,
        False if the point is on water
        
        coord is a (long, lat) location
        
        """
        raise NotImplimentedError

   def in_water(self, coord):
        """
        returns a Boolean result:
        
        True if the point is in the water,
        False if the point is on land (or off map?)
        
        coord is a (long, lat) location
        
        """
        raise NotImplimentedError
            
    def allowable_spill_position(self, coord):
        """
        returns a Boolean result:
        
        True if the point is an allowable spill position
        False if the point is not an allowable spill position

        (Note: it could be either off the map, or in a location that spills
        aren't allowed)
        
        coord is a (long, lat) location
        
        """
        raise NotImplimentedError

class RasterMap(LandWaterMap):

    """
    A land water map implemented as a raster
    """

    water = 0
    land  = 1
    spillable_area = 2
    # others....
    
        
    def __init__(self, refloat_halflife, bitmap_array, projection, bounding_box):
        """
        create a new RasterMap
        
        refloat_halflife is the halflife for refloating
        
        This is assumed to be the same everywhere at this point
        
        bitmap_array is a numpy array that stores the land-water map
        
        projection is the projection object -- used to conver from lat-long to pixels in the array
        
        bounding box is the bounding box of the map -- may not match the array -- if the map is larger than the land.
    
        self.bitmap = bitmap_array
        self.bounding_box = hazpy.geometry.BBox.BBox(bounding_box)
        self.refloat_halflife = refloat_halflife

    def pixel_on_map(self, pixel_coord):
        """
        returns True is the pixel location is on the map
        
        note: should this support no-rectangluar maps?
        """
        
        if self.bounding_box.PointInside(pixel_coord):
            return True
        else:
            return False

    def on_land(self, coord):
        """
        returns True is on land
        
        coord is (long, lat) location
        
        """
        return not self.in_water(coord)

    def on_land_pixel(self, coord):
        return not self.in_water_pixel(coord)
        
    def in_water(self, coord):
        coord = self.to_pixel(coord)
        return self.in_water_pixel(coord)
    
    def in_water_pixel(self, coord):
        if not self.on_map(coord):
            return False
        try:
            return self.bitmap[coord[0], coord[1]]
        except IndexError:
            # Note: this could be off map, which may be a different thing than on land....
            return False

    def allowable_spill_position(self, coord):
        """
        returns true is the spill position is in teh allowable spill area
        
        This may not be the same as in_water!
        """
        ##fixme: add check for allowable spill area -- may not be all water!
        return self.in_water(coord)
        
