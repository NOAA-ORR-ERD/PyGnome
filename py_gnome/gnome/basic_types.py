#!/usr/bin/env python

"""
basic_types.py

The python version of the various type definitions used

Imports all the symbols from cy_basic_types.pyx

Adds some for Python-only use

"""

import numpy as np

from cy_gnome.cy_basic_types import * # pull everything from the cython code


mover_type = np.float64
world_point_type = np.float64
windage_type = np.float64
water_current_type = np.float64

datetime_value_2d = np.dtype([('time', 'datetime64[s]'), 
                              ('value', mover_type,(2,))], align=True)

# enums that are same as C++ values are defined in cy_basic_types
# Define enums that are independent of C++ here so we don't have to recompile code
"""
Define enum for type of wind_source
"""
wind_datasource = enum(undefined=0, file=1, manual=2, nws=3, buoy=4)

#----------------------------------------------------------------
# Mirror C++ structures, following are used by cython code to access C++ methods/classes
seconds = np.int # model time is going to be given in seconds
world_point = np.dtype([('long', world_point_type),
                        ('lat', world_point_type),
                        ('z', world_point_type)],
                       align=True)
velocity_rec       = np.dtype([('u', np.double), ('v', np.double),], align=True)
time_value_pair    = np.dtype([('time', seconds), ('value', velocity_rec),], align=True)
ebb_flood_data    = np.dtype([('time', seconds), ('speedInKnots', np.double),('type',np.short),], align=True)
tide_height_data    = np.dtype([('time', seconds), ('height', np.double),('type',np.int16),], align=True)

# This 2D world point is just used by shio and Cats at present
w_point_2d = np.dtype([('long', world_point_type),
                       ('lat', world_point_type)])                           

# In the C++ TypeDefs.h, the enum type for LEStatus is defined as a short
# this is also consistent with the definition in type_defs.pxd .. define it here to keep things consistent
status_code_type = np.int16

id_type = np.uint16
