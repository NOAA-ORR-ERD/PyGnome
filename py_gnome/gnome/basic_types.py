"""
basic_types.py

The python version of the various type definitions used

Imports all the symbols from cy_basic_types.pyx

Adds some for Python-only use

"""

import sys

import numpy as np

# using the Py3 enum type
from enum import IntEnum

# pull everything from the cython code
from .cy_gnome import cy_basic_types as cbt

# in lib_gnome, the coordinate systems used (r-theta, uv, etc)
# are called ts_format, which is not a very descriptive name.
# the word 'format' can mean a lot of different things depending on
# what we are talking about.  So we try to be a bit more specific here.
# pull the enums from the cython version

ts_format = cbt.ts_format
oil_status = cbt.oil_status
spill_type = cbt.spill_type

seconds = cbt.seconds

# this is a mapping of oil_status code to the meaningful name:
oil_status_map = {num: name for name, num in oil_status.__members__.items()}


# Here we customize what a numpy 'long' type is....
# We do this because numpy does different things with a long
# that can mismatch what Cython does with the numpy ctypes.
# NOTE: This is all so we can create numpy arrays that are compatible
#       with the structs in the C++ code -- most of which
#       use the old-style C types, e.g. int, short, long.
# In particular, long is defined differently on Windows and Linux
# so we need to make sure that we are using the right type on each platform
# note: numpy no longer HAS numpy.long or numpy.int
#       there is a numpy.compat.long, which maps to int (which is the Python int type)
# however, I *think* that Python uses a C long for int,
# and numpy figures it out for itself at run time, so I think we can
# just do that:

np_long = int

# THE REST OF THIS IS OUT OF DATE:
# on OSX 64:
# - numpy identifies np.long as int64, which is a Cython ctype 'long'
# on OSX 32:
# - numpy identifies np.long as int64, which is a Cython ctype 'long long'
# on Win32:
# - numpy identifies np.long as int64, which is a Cython ctype 'long long',
# on Win64:
# - unknown what numpy does
# - Presumably an int64 would be a ctype 'long' ???
# we should clean this up! -- on Python3, I think np.compat.long == int

########
# NOTE: all this below was essentially simply setting np_long to int
#  on all platforms anyway.
# IF we have future problems, it might be as simple as:
# if sys.platform == 'win32':
#     np_long = np.int32
# elif sys.platform in ('darwin', 'linux2', 'linux'):
#     np_long = np.int64
########

# if sys.platform == 'win32':
#     np_long = np.int
# elif sys.platform in ('darwin', 'linux2', 'linux'):
#     if sys.maxsize > 2147483647:
#         np_long = np.compat.long
#     else:
#         np_long = int
# else:
#     # untested platforms will just default to what numpy thinks it is.
#     np_long = np.compat.long

mover_type = np.float64
world_point_type = np.float64
windage_type = np.float64
water_current_type = np.float64

# value has two components: (u, v) or (r, theta) etc
datetime_value_2d = np.dtype([('time', 'datetime64[s]'),
                              ('value', mover_type, (2,))], align=True)

# value has one component: (u,)
# convert from datetime_value_1d to time_value_pair by setting 2nd component
# of value to 0.0
datetime_value_1d = np.dtype([('time', 'datetime64[s]'),
                              ('value', mover_type, ())], align=True)


# enums that are same as C++ values are defined in cy_basic_types
# Define enums that are independent of C++ here so we
# don't have to recompile code
# fixme: that seems dangerous!
class wind_datasources(IntEnum):
    undefined = 0
    file = 1
    manual = 2
    nws = 3
    buoy = 4


# Define an enum for weathering status. The numpy array will contain np.uint8
# datatype. Can still define 2 more flags as 2**6, 2**7
# These are bit flags
class fate(IntEnum):
    """
    An enum for weathering status. The numpy array will contain np.uint8
    datatype. Can still define 2 more flags as 2**6, 2**7
    These are bit flags
    """
    non_weather = 1,
    surface_weather = 2,
    subsurf_weather = 4,
    skim = 8,
    burn = 16,
    disperse = 32,  # marked for chemical_dispersion


class numerical_methods(IntEnum):
    Euler = 0
    RK2 = 1
    RK4 = 2

# ----------------------------------------------------------------
# Mirror C++ structures, following are used by cython code
# to access C++ methods/classes

world_point = np.dtype([('long', world_point_type),
                        ('lat', world_point_type),
                        ('z', world_point_type)],
                       align=True)
velocity_rec = np.dtype([('u', np.double),
                         ('v', np.double)],
                        align=True)
time_value_pair = np.dtype([('time', seconds),
                            ('value', velocity_rec)],
                           align=True)
ebb_flood_data = np.dtype([('time', seconds),
                           ('speedInKnots', np.double),
                           ('type', np.short)],
                          align=True)
tide_height_data = np.dtype([('time', seconds),
                             ('height', np.double),
                             ('type', np.int16)],
                            align=True)

# This 2D world point is just used by shio and Cats at present
w_point_2d = np.dtype([('long', world_point_type),
                       ('lat', world_point_type)])

long_point = np.dtype([('long', np_long),
                       ('lat', np_long)],
                      align=True)

triangle_data = np.dtype([('v1', np_long), ('v2', np_long), ('v3', np_long),
                          ('n1', np_long), ('n2', np_long), ('n3', np_long)],
                         align=True)

cell_data = np.dtype([('cell_num', np_long),
                      ('top_left', np_long), ('top_right', np_long),
                      ('bottom_left', np_long), ('bottom_right', np_long)],
                     align=True)

# In the C++ TypeDefs.h, the enum type for LEStatus is defined as a short
# this is also consistent with the definition in type_defs.pxd ..
# define it here to keep things consistent

status_code_type = np.int16

# id_type is dtype for numpy array for 'spill_num'.
# This is NOT currently passed to C++
id_type = np.uint16

# ------------------------------------------------
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
