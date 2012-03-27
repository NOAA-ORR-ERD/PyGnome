 #!/usr/bin/env python
 
 # I don't have documentation strings
 
 # if __name__ == __???___:
 #
 #
 
import sys
import os
import numpy as np
import random
from PIL import Image, ImageDraw

from gnome.utilities import map_canvas
from hazpy.file_tools import haz_files
from hazpy.geometry import polygons

class gnome_map(map_canvas.MapCanvas):
    """
        Basics pyGNOME color bitmap.
        (End-user visualization.)
    """     
    def __init__(self, image_size, bna_filename, color_mode='RGB'):
        """
            Initializes color map attributes. Calls on parent class initialization 
            method in order to handle projection scaling.
        """
        map_canvas.MapCanvas.__init__(self, image_size, projection=map_canvas.FlatEarthProjection, mode=color_mode)
        self.polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
        self.filename = bna_filename        
        self.draw_land(self.polygons)

    def to_pixel(self, coord):
        """ Projects a (lon, lat) tuple onto the bitmap, and returns the resultant tuple. """
        coord = tuple(self.projection.to_pixel(np.array((coord[0], coord[1]))))
        coord = (int(coord[0]), int(coord[1]))
        return coord
        
    def to_pixel_array(self, coords):
        """ 
            Projects an array of (lon, lat) tuples onto the bitmap, and modifies it in 
            place to hold the corresponding projected values.
        """
        coords['p_long'] -= self.projection.center[0]
        coords['p_lat'] -= self.projection.center[1]
        coords['p_long'] *= self.projection.scale[0]
        coords['p_lat'] *= self.projection.scale[1]
        coords['p_long'] += self.projection.offset[0]
        coords['p_lat'] += self.projection.offset[1]
        coords['p_long'] = np.round(coords['p_long']).astype(np.int)
        coords['p_lat'] = np.round(coords['p_lat']).astype(np.int)
    
    
    def _type(self):
        """ This requires an explanation. """
        return ' color bitmap'
        
class lw_map(gnome_map):

    """land-water bitmap."""
        
    background_color = 0
    lake_color = 0
    land_color = 1
    
    def __init__(self, image_size, bna_filename, refloat_halflife, color_mode='1'):
        """
            Initializes land-water map attributes. Calls on parent class initialization 
            method in order to handle projection scaling. Caches its bounding_box so that
            it doesn't need to be computed repeatedly.
        """
        gnome_map.__init__(self, image_size, bna_filename, color_mode)
        self.bounding_box = self.polygons.bounding_box
        self.refloat_halflife = refloat_halflife
        self.spills = []

    def _type(self):
        """ Returns the map type. (Either 'color' or 'land-water'.) """
        return ' land-water bitmap'
        
    def on_map(self, pixel_coord):
        """ 
            Given a tuple in pixel coordinates, determines whether the position is on the map.
            It is actually not behaving correctly at the moment: the map bounds and the bounding box
            may not necessarily coincide. Needs fixing!
        """
        bounding_box = self.bounding_box
        if pixel_coord[0] > bounding_box[0][0] and pixel_coord[0] < bounding_box[1][0] and pixel_coord[1] > bounding_box[0][1] and pixel_coord[1] < bounding_box[1][1]:
            return True
        return False

    def on_land(self, coord):
        """ Given lat-lon coordinates, determines whether the position is on land. """
        return not self.in_water(coord)

    def on_land_pixel(self, coord):
        """ Given a tuple in pixel coordinates, determines whether the position is on land. """
        return not self.in_water_pixel(coord)
        
    def in_water(self, coord):
        """ Given lat-lon coordinates, determines whether the position is in water. """
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
        """ Given a tuple in pixel coordinates, determines whether the position is in water. """
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
        """ Determines whether a position given in lat-lon coordinates is an allowable spill location. """
        return self.in_water(coord)
        
    def set_spill(self, coord, num_particles, release_time):
        """ 
            Sets a spill.
        ++args:
            coord: (lon, lat)
            release_time in seconds.
        """
        if not self.allowable_spill_position(coord):
            print  "spill " + str(dict((('position', coord), ('num_particles', num_particles), ('release_time', release_time)))) + " ignored."
        else:
            self.spills += [(coord, num_particles, release_time)]

    def beach_element(self, p, lwp):
        """ 
            Beaches an element that has been landed.
        ++args:
            p: current position (see basic_types.world_point dtype)
            lwp: last water position (see basic_types.world_point dtype)
        """
        in_water = self.in_water
        displacement = ((p['p_long'] - lwp['p_long']), (p['p_lat'] - lwp['p_lat']))
        while not in_water((p['p_long'], p['p_lat'])):
            displacement = (displacement[0]/2, displacement[1]/2)
            p['p_long'] = lwp['p_long'] + displacement[0]
            p['p_lat'] = lwp['p_lat'] + displacement[1]