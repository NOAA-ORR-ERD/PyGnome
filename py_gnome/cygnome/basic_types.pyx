"""
cython file used to store all the type info for GNOME.

Some hard-coded here, some pulled form C++ headers, etc.
"""

import cython

## pull stuff in from the C++ headers
include "type_defs.pxi"

## Various Python-only type definitions

## Package wide type definitions
import numpy as np

# Basic types. Should each have a corresponding, equivalent type in the C++ (type_defs.pxi)

world_point_type = np.float64
world_point = np.dtype([('p_long', world_point_type),
                        ('p_lat', world_point_type)],
                       align=True)
world_point_3d = np.dtype([('p', world_point),
                           ('z', np.float64)],
                          align=True)

world_rect = np.dtype([('lo_long', np.long),
                       ('lo_lat', np.long),
                       ('hi_long', np.long),
                       ('hi_lat', np.long)],
                      align=True)

#le_rec = np.dtype([('le_units', np.int), ('le_key', np.int), ('le_custom_data', np.int), 
#                      ('p', world_point), ('z', np.double), ('release_time', np.uint), 
#                      ('age_in_hrs_when_released', np.double), ('clock_ref', np.uint), 
#                      ('pollutant_type', np.short), ('mass', np.double), ('density', np.double), 
#                      ('windage', np.double), ('droplet_size', np.int), ('dispersion_status', np.short), 
#                      ('rise_velocity', np.double), ('status_code', np.short), ('last_water_pt', world_point), ('beach_time', np.uint)], align=True)

## fixme -- I could be put in the wind_mover code...
##   it seems they are only relevent to the wind mover.
windage_type = np.float64
wind_uncertain_rec = np.dtype([('randCos', np.float32), ('randSin', np.float32),], align=True)
le_uncertain_rec   = np.dtype([('downStream', np.float32), ('crossStream', np.float32),], align=True)
velocity_rec       = np.dtype([('u', np.double), ('v', np.double),], align=True)
time_value_pair    = np.dtype([('time', np.uint32), ('value', velocity_rec),], align=True)

status_code_type = np.int16 # does it matter, as long as it's an int type???
