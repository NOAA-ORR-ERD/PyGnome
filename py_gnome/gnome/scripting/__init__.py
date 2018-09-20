"""
Scripting package for GNOME with assorted utilities that make it easier to
write scripts.

The ultimate goal is to be able to run py_gnome for the "common" use cases
with only functions available in this module

Helper functions are imported from various py_gnome modules
(spill, environment, movers etc).

contrary to the usual practive, this module is designed to by used like so:

from gnome.scripting import *

Then you will have easy access to most of the stuff you need to write simple py_gnome scripts.py_gnome

"""


__all__ = ['constant_wind',
           'point_line_release_spill',
           'surface_point_line_spill',
           'subsurface_plume_spill',
           'grid_spill',
           "make_images_dir",
           "remove_netcdf",
           'seconds',
           'hours',
           'minutes',
           'days',
           'weeks',
           'Model',
           'set_verbose',
           'Renderer',
           'NetCDFOutput',
           'KMZOutput',
           'MapFromBNA',
           'GridCurrent',
           'RandomMover',
           'WindMover',
           ]

import gnome
from gnome.model import Model

from .utilities import *
from time_utils import *

from gnome.environment.wind import constant_wind

from gnome.spill.spill import (point_line_release_spill,
                               surface_point_line_spill,
                               subsurface_plume_spill,
                               grid_spill,
                               )

from gnome.movers.wind_movers import (constant_wind_mover,
                                      wind_mover_from_file,
                                      )

from gnome.outputters import Renderer, NetCDFOutput, KMZOutput
from gnome.map import MapFromBNA
from gnome.environment import GridCurrent
from gnome.movers import RandomMover, WindMover


def set_verbose(log_level='info'):
    """
    Set the logging system to dump to the console --
    you can see muchmore what's going on with the model
    as it runs

    :param log_level='info': the level you want your log to show. options are,
                             in order of importance: "debug", "info", "warning",
                             "error", "critical".

    You will only get the logging messages at or above the level you set.
    Set to "debug" for everything.
    """
    gnome.initialize_console_log(log_level)


