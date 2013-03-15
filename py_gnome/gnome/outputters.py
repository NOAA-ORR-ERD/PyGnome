#!/usr/bin/env python

"""
outputters.py

module to define classes for GNOME output:
  - image writting
  - saving to netcdf
  - saving to other formats ?

"""
import os

from gnome.utilities.map_canvas import  MapCanvas
from gnome.utilities import projections
from gnome.utilities.file_tools import haz_files


class Outputter(object):
    """
    base class for all outputters
    """

    def write_step(self, ):
        """
        called by the model at the end of each time step

        write the ouput here
        """
        pass

    def prepare_for_model_run(self):
        """
        This method gets called by the model at the beginning of a new run. 
        Do what you need to do to prepare.
        """
        pass

    def prepare_for_model_step(self):
        """
        This method gets called by the model at the beginning of each time step. 
        Do what you need to do to prepare for a new model step
        """
        pass

    def model_step_is_done(self):
        """
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc.
        """
        pass 

    def rewind(self):
        """
        called by model.rewind()

        do what needs to be done to reset the outputter
        """
        pass 

class ImageOutputter(Outputter, MapCanvas):
    """
    ImageOutputter

    class that writes map images for GNOME results:
        writes the frames for the LE "movies"
    """
    def __init__(self, mapfile, images_dir, size = (800,600), projection_class=projections.FlatEarthProjection):
        """
        Init the image outputter.
        """
        MapCanvas.__init__(self, size, projection_class)
        #set up the canvas
        polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
        self.set_land(polygons)

        self.images_dir = images_dir
    
    def prepare_for_model_run(self):
        self.draw_background()
        self.save_background(os.path.join(images_dir, "background_map.png"))

    def write_step(self, step_num, cache):
        """
        Render the map image, according to current parameters.

        :param step_num: the current step number of the model.
        """

        filename = os.path.join(self.images_dir, 'foreground_%05i.png'%step_num)

        self.create_foreground_image()

        # pull the data from cache:
        for sc in cache.load_timestep(step_num).items():
            self.draw_elements(sc)

        self.save_foreground(filename)

        return filename

