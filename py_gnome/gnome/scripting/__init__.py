"""
scripting package for gnome

assorted utilities that make it easier to write scripts to automate gnome

"""
from .utilities import *
from gnome.environment.wind import constant_wind
from gnome.spill.spill import point_line_release_spill
from gnome.spill.spill import surface_point_line_spill
from gnome.movers.wind_movers import constant_wind_mover
from gnome.movers.wind_movers import wind_mover_from_file
