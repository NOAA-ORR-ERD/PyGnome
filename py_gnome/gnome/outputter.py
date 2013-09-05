#!/usr/bin/env python

"""
outputters.py

module to define classes for GNOME output:
  - base class
  - saving to netcdf
  - saving to other formats ?

"""

import os

from gnome.utilities.map_canvas import MapCanvas
from gnome.utilities import projections
from gnome.utilities.file_tools import haz_files


class Outputter(object):

    """
    base class for all outputters
    """

    def write_output(self, step_num):
        """
        called by the model at the end of each time step

        write the ouput here
        """

        pass

    def prepare_for_model_run(self, **kwargs):
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


