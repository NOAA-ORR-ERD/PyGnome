import os
from os.path import basename
import glob
import copy
import zipfile
import numpy as np

from colander import SchemaNode, String, drop

from gnome.persist import base_schema, class_from_objtype

from . import Renderer
from gnome.utilities.map_canvas_gd import MapCanvas
from gnome.utilities.serializable import Field
from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections

from gnome.basic_types import oil_status


class Animation(Renderer):
    def __init__(self, *args, **kwargs):
        Renderer.__init__(self, *args, **kwargs)

    def clean_output_files(self):
        # clear out the output dir:
        try:
            os.remove(os.path.join(self.output_dir,
                                   self.background_map_name))
        except OSError:
            # it's not there to delete..
            pass

        anim_file = os.path.join(self.output_dir, self.animation_filename)
        os.remove(anim_file)

    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the renderer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define ``**kwargs`` and pass into the
        base class

        In this case, it draws the background image and clears the previous
        images. If you want to save the previous images, a new output dir
        should be set.
        """
        super(Renderer, self).prepare_for_model_run(*args, **kwargs)

        self.clean_output_files()

        self.draw_background()
        
        self.save_background(os.path.join(self.output_dir,
                                          self.background_map_name)