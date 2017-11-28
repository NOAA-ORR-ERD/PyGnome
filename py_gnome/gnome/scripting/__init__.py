"""
Scripting package for GNOME with assorted utilities that make it easier to
write scripts.

The ultimate goal is to be able to run py_gnome for the "common" use cases
with only functions available in this module

Helper functions are imported from various py_gnome modules
(spill, environment, movers etc).

"""
from .utilities import *

from gnome.environment.wind import constant_wind

from gnome.spill.spill import (point_line_release_spill,
                               surface_point_line_spill,
                               subsurface_plume_spill,
                               grid_spill,
                               )
from gnome.movers.wind_movers import (constant_wind_mover,
                                      wind_mover_from_file,
                                      )

