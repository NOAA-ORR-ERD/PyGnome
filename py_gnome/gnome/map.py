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

class lw_map(gnome_map):

    """land-water bitmap."""
        
    background_color = 0
    lake_color = 0
    land_color = 1
    
    def __init__(self, image_size, bna_filename, refloat_halflife, color_mode='1'):
        gnome_map.__init__(self, image_size, bna_filename, color_mode)
        self.refloat_halflife = refloat_halflife
        self.spills = []
    
    def __del__(self):
        pass
        
    def get_bounds(self):
        return self.polygons.bounding_box

    def on_map(self, coord):
        return True

    def on_land(self, coord):
        return (self.on_map(coord) and not self.in_water(coord))


    def in_water(self, coord):
        if not self.on_map(coord):
            return False
        coord = self.to_pixel(coord)
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
