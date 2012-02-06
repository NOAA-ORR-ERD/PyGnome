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

sys.path[len(sys.path):] = [os.environ['HOME']+'/Workspace/GNOME2']

from utilities import map_canvas
from hazpy.file_tools import haz_files
from hazpy.geometry import polygons

class map(map_canvas.MapCanvas):
    
    """
    Inherits MapCanvas.
    Also wraps c_map. (Not Yet)
    
    """
    background_color = 0
    lake_color = 0
    land_color = 1
    
    def __init__(self, image_size, bna_filename):
        map_canvas.MapCanvas.__init__(self, image_size, projection=map_canvas.FlatEarthProjection, mode='1')
        self.polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
        self.filename = bna_filename
        self.draw_land(self.polygons)
        self.spills = []
    
    def __del__(self):
        pass        

    def to_pixel(self, coord):
        return self.projection.to_pixel((coord,))[0]

    def get_bounds(self):
        return self.polygons.bounding_box
    
    def on_map(self, coord):
        return True

    def on_land(self, coord):
        coord = self.to_pixel(coord)
        return (self.on_map(coord) and not self.in_water(coord))

    def allowable_spill_position(self, coord):
        coord = self.to_pixel(coord)
        return self.in_water(coord)
        
    def in_water(self, coord):
        if not self.on_map(coord):
            return False
        coord = self.to_pixel(coord)
        chrom = self.image.getpixel(coord)
        if not chrom:
            return True
        else:
            return False

    def agitate_particles(self, time_step, spill):
         pass

    def set_spill(self, coord, num_particles, release_time):
        if not self.allowable_spill_position(coord):
            return False
        self.spills += [(coord, num_particles, release_time)]
        
