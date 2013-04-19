#!/usr/bin/env python

"""
renderer.py

module to hold all teh map rendering code.

"""

import os, glob

import datetime

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
    background_map_name = 'background_map.png'
    foreground_filename_format = 'foreground_%05i.png'
    foreground_filename_glob =   'foreground_?????.png'

    def __init__(self,
                 mapfile,
                 images_dir,
                 size = (800,600),
                 projection_class=projections.FlatEarthProjection,
                 cache = None):

        """
        Init the image renderer.
        """
        #set up the canvas
        polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
        MapCanvas.__init__(self,
                           size,
                           land_polygons=polygons,
                           projection_class=projection_class)

        self.images_dir = images_dir
    
        self.cache = cache
        self.last_filename = ''

    def prepare_for_model_run(self, cache=None):
        """
        prepares the renderer for a model run.

        In this case, it draws the background image and clears the previous images

        If you want to save the previous images, a new output dir should be set.

        :param cache=None: Sets the cache object to be used for the data.
                           If None, it will use teh one already set up.

        """
        if cache is not None:
            self.cache = cache

        self.clear_output_dir()

        self.draw_background()
        self.save_background(os.path.join(self.images_dir, self.background_map_name))

    def clear_output_dir(self):
        # clear out output dir:
        # don't need to do this -- it will get written over.
        
        try:
            os.remove(os.path.join(self.images_dir, self.background_map_name))
        except OSError: # it's not there to delete..
            pass

        foreground_filenames = glob.glob(os.path.join(self.images_dir, self.foreground_filename_glob))
        for name in foreground_filenames:
            os.remove(name)

    def write_output(self, step_num):
        """
        Render the map image, according to current parameters.

        :param step_num: the model step number you want rendered.

        :returns: A dict of info about this step number:
                   'step_num': step_num
                   'image_filename': filename 
                   'time_stamp': time_stamp # as ISO string

        """

        filename = os.path.join(self.images_dir, self.foreground_filename_format%step_num)

        self.create_foreground_image()

        # pull the data from cache:
        for sc in self.cache.load_timestep(step_num).items():
            self.draw_elements(sc)

        # get the timestamp:
        time_stamp = sc['current_time_stamp'].item().isoformat()
        self.save_foreground(filename)

        self.last_filename = filename

        return {'step_num': step_num,
                'image_filename': filename,
                'time_stamp': time_stamp}

