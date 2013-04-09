#!/usr/bin/env python

"""
renderer.py

module to hold all teh map rendering code.

"""

import os

from gnome.outputter import Outputter
from gnome.utilities.map_canvas import  MapCanvas
from gnome.utilities import projections
from gnome.utilities.file_tools import haz_files



class Renderer(Outputter, MapCanvas):
    """
    Map Renderer

    class that writes map images for GNOME results:
        writes the frames for the LE "movies", etc.

    """
    def __init__(self, mapfile, images_dir, size = (800,600), projection_class=projections.FlatEarthProjection):
        """
        Init the image renderer.
        """
        MapCanvas.__init__(self, size, projection_class)
        #set up the canvas
        polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
        self.set_land(polygons)

        self.images_dir = images_dir
    
        self.cache = None
        self.last_filename = ''

    def prepare_for_model_run(self, cache):

        self.cache = cache

        self.draw_background()
        self.save_background(os.path.join(self.images_dir, "background_map.png"))

    def write_output(self, step_num):
        """
        Render the map image, according to current parameters.

        :param step_num: the current step number of the model.
        """

        filename = os.path.join(self.images_dir, 'foreground_%05i.png'%step_num)

        self.create_foreground_image()

        # pull the data from cache:
        for sc in self.cache.load_timestep(step_num).items():
            self.draw_elements(sc)

        self.save_foreground(filename)

        self.last_filename = filename

        return None

