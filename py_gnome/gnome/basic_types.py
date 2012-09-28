#!/usr/bin/env python

"""
basic_types.py

The python version of the various type definitions used

Imports all the symbols from cy_basic_types.pyx

Adds some for Python-only use

"""

import datetime
import numpy as np


from cy_gnome.cy_basic_types import * # pull everything from the cython code

## Various Python-only type definitions


## Time stuff:
EPOCH = datetime.datetime(1970, 1, 1)
def dt_to_epoch(dt):
    """
    converts a python datetime to the epoch used in the GNOME C++ code
    """
    return (dt - EPOCH).total_seconds()


# Basic types. Should each have a corresponding, equivalent type in the C++ (type_defs.pxi)

mover_type = np.float64 # the type used by the movers to do their calculations
                        # data should genreally be passed in and stored in this type

world_point_type = np.float64

world_point = np.dtype([('long', world_point_type),
                        ('lat', world_point_type),
                        ('z', world_point_type)],
                       align=True)

#world_point_3d = np.dtype([('p', world_point),
#                           ('z', np.float64)],
#                          align=True)

#world_rect = np.dtype([('lo_long', np.long),
#                       ('lo_lat', np.long),
#                       ('hi_long', np.long),
#                       ('hi_lat', np.long)],
#                      align=True)

#le_rec = np.dtype([('le_units', np.int), ('le_key', np.int), ('le_custom_data', np.int), 
#                      ('p', world_point), ('z', np.double), ('release_time', np.uint), 
#                      ('age_in_hrs_when_released', np.double), ('clock_ref', np.uint), 
#                      ('pollutant_type', np.short), ('mass', np.double), ('density', np.double), 
#                      ('windage', np.double), ('droplet_size', np.int), ('dispersion_status', np.short), 
#                      ('rise_velocity', np.double), ('status_code', np.short), ('last_water_pt', world_point), ('beach_time', np.uint)], align=True)

## fixme -- I could be put in the wind_mover code...
##   it seems they are only relevent to the wind mover.

windage_type = np.float64

seconds = np.uint32 # model time is going to be given in seconds

wind_uncertain_rec = np.dtype([('randCos', np.float32), ('randSin', np.float32),], align=True)
le_uncertain_rec   = np.dtype([('downStream', np.float32), ('crossStream', np.float32),], align=True)
velocity_rec       = np.dtype([('u', np.double), ('v', np.double),], align=True)
time_value_pair    = np.dtype([('time', seconds), ('value', velocity_rec),], align=True)
date_rec           = np.dtype([('year', np.short),
                               ('month', np.short),
                               ('day', np.short),
                               ('hour', np.short),
                               ('minute', np.short),
                               ('second', np.short), 
                               ('dayOfWeek', np.short),], align=True)

status_code_type = np.int16 # does it matter, as long as it's an int type???
