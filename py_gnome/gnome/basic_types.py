"""
basic_types.py

The python version of the various type definitions used

Imports all the symbols from cy_basic_types.pyx

Adds some for Python-only use

"""

import sys

import numpy
np = numpy

from cy_gnome.cy_basic_types import *  # pull everything from the cython code

# Here we customize what a numpy 'long' type is....
# We do this because numpy does different things with a long
# that can mismatch what Cython does with the numpy ctypes.
# on OSX 64:
# - numpy identifies np.long as int64, which is a Cython ctype 'long'
# - this is fine
# on OSX 32:
# - numpy identifies np.long as int64, which is a Cython ctype 'long long'
# - this is a problem
# on Win32:
# - numpy identifies np.long as int64, which is a Cython ctype 'long long',
# - ***this mismatches the ctype 'long'
# on Win64:
# - unknown what numpy does
# - Presumably an int64 would be a ctype 'long' ???

if sys.platform == 'win32':
    np_long = np.int
elif sys.platform == 'darwin':
    if sys.maxint > 2147483647:
        np_long = np.long
    else:
        np_long = np.int
else:

    # untested platforms will just default

    np_long = np.long

mover_type = np.float64
world_point_type = np.float64
windage_type = np.float64
water_current_type = np.float64

# value has two components: (u, v) or (r, theta) etc
datetime_value_2d = np.dtype([('time', 'datetime64[s]'),
    ('value', mover_type, (2, ))], align=True)

# value has one component: (u,)
# convert from datetime_value_1d to time_value_pair by setting 2nd component
# of value to 0.0
datetime_value_1d = np.dtype([('time', 'datetime64[s]'),
    ('value', mover_type, (1,))], align=True)

# enums that are same as C++ values are defined in cy_basic_types
# Define enums that are independent of C++ here so we
# don't have to recompile code

wind_datasource = enum(undefined=0, file=1, manual=2, nws=3, buoy=4)

# ----------------------------------------------------------------
# Mirror C++ structures, following are used by cython code to
# access C++ methods/classes

world_point = np.dtype([('long', world_point_type), ('lat',
                       world_point_type), ('z', world_point_type)],
                       align=True)
velocity_rec = np.dtype([('u', np.double), ('v', np.double)],
                        align=True)
time_value_pair = np.dtype([('time', seconds), ('value',
                           velocity_rec)], align=True)
ebb_flood_data = np.dtype([('time', seconds), ('speedInKnots',
                          np.double), ('type', np.short)], align=True)
tide_height_data = np.dtype([('time', seconds), ('height', np.double),
                            ('type', np.int16)], align=True)

# This 2D world point is just used by shio and Cats at present

w_point_2d = np.dtype([('long', world_point_type), ('lat',
                      world_point_type)])

# In the C++ TypeDefs.h, the enum type for LEStatus is defined as a short
# this is also consistent with the definition in type_defs.pxd ..
# define it here to keep things consistent

status_code_type = np.int16

# id_type is dtype for numpy array for 'spill_num'. This is NOT currently passed to C++

id_type = np.uint16


#------------------------------------------------
# NOTE: This is only used to test that the python time_utils
# converts from date to sec and sec to date in the same way
# as the C++ code. Currently, cy_helpers defines the CyDateTime
# class which is merely used for testing the time_utils conversions
# test_cy_helpers.TestCyDateTime class contians these tests
date_rec = np.dtype([
        ('year', np.short),
        ('month', np.short),
        ('day', np.short),
        ('hour', np.short),
        ('minute', np.short),
        ('second', np.short),
        ('dayOfWeek', np.short),
        ], align=True)
