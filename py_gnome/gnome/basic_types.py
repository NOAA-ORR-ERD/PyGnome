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

datetime_r_theta   = np.dtype([('time', np.datetime64), 
                               ('value',mover_type,(2,))], align=True)
datetime_value_pair= np.dtype([('time', np.datetime64), 
                               ('value', mover_type,(2,))], align=True)

#----------------------------------------------------------------
# Mirror C++ structures, following are used by cython code
seconds = np.uint32 # model time is going to be given in seconds
world_point = np.dtype([('long', world_point_type),
                        ('lat', world_point_type),
                        ('z', world_point_type)],
                       align=True)
velocity_rec       = np.dtype([('u', np.double), ('v', np.double),], align=True)
time_value_pair    = np.dtype([('time', seconds), ('value', velocity_rec),], align=True)

# only used in test_cy_date_time to validate time_utils.date_to_sec functionality
date_rec           = np.dtype([('year', np.short),
                               ('month', np.short),
                               ('day', np.short),
                               ('hour', np.short),
                               ('minute', np.short),
                               ('second', np.short), 
                               ('dayOfWeek', np.short),], align=True)

status_code_type = np.short # In the C++ TypeDefs.h, the enum type for LEStatus is defined as a short
                            # this is also consistent with the definition in type_defs.pxd .. define it here to keep things consistent
                            


