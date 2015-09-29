"""
scripting package for gnome

assorted utilities that make it easier to write scripts to automate gnome

"""
from .utilities import *
from gnome.environment.wind import constant_wind
from gnome.spill import point_line_release_spill
from gnome.movers.wind import constant_wind_mover
from gnome.movers.wind import wind_mover_from_file
