"""
cython file used to store all the type info for GNOME.

Pulled from type_defs.pxi -- i.e pulled from C++ headers, etc.

"""

# Use the "new" Py3 Enum type
from enum import IntEnum

# from gnome.utilities.enum import IntEnum

# pull stuff in from the C++ headers
from gnome.cy_gnome.type_defs cimport *


'''
def enum(**enums):
    """
    Usage: x = enum(a=1, b=2, c=3)
           x.a = 1, x.b = 2, x.c = 3
           x._attr = ['a','b','c'], x._int = [ 1, 2, 3]

    - Returns a new type called Enum whose attributes are given by the input
      in 'enums'
    - also append two more attributes called:
      _attr - contains the list of keywords in the input
      _int - contains the list of int values for this enum
    These are in the same order, and can be helpful for error checking, etc
    Make sure your enum name here is NOT THE SAME as in the C++ code or it
    will not build!
    """
    # append keys and int to dict
    enums.update({'_attr': enums.keys(), '_int': enums.values()})

    return type('Enum', (), enums)
'''

"""
LE Status as an enum type
"""
class oil_status(IntEnum):
    """
    maps to the C enum
    """
    not_released = OILSTAT_NOTRELEASED
    in_water = OILSTAT_INWATER
    on_land = OILSTAT_ONLAND
    off_maps = OILSTAT_OFFMAPS
    evaporated = OILSTAT_EVAPORATED
    to_be_removed = OILSTAT_TO_BE_REMOVED
    on_tideflat = OILSTAT_ON_TIDEFLAT


class numerical_methods(IntEnum):
    euler = EULER,
    rk4 = RK4

"""
disperse status as an enum type
"""
# disp_status = dict(dont_disperse = DONT_DISPERSE,
#                    disperse = DISPERSE,
#                    have_dispersed = HAVE_DISPERSED,
#                    disperse_nat = DISPERSE_NAT,
#                    have_dispersed_nat = HAVE_DISPERSED_NAT,
#                    evaporate = EVAPORATE,
#                    have_evaporated = HAVE_EVAPORATED,
#                    remove = REMOVE,
#                    have_removed = HAVE_REMOVED)
#

"""
SpillType {FORECAST_LE = 1, UNCERTAINTY_LE = 2};
"""
class spill_type(IntEnum):
    forecast = FORECAST_LE
    uncertainty = UNCERTAINTY_LE


"""
Contains enum type for the timeseries (ts) either given directly or
read from datafile, by OSSMTimeValue.
For instance, a standard wind file would contain magnitude and direction info
  ts_format.magnitude_direction = 5,

'r-theta' is another alias for this so,
  ts_format.r_theta = 5


It could also contain uv info. Tides would contain uv with v == 0
Hydrology file would also contain uv format
from TypeDefs.h:
*   M19REALREAL = 1,
    M19HILITEDEFAULT = 2
    M19MAGNITUDEDEGREES = 3
    M19DEGREESMAGNITUDE = 4
*   M19MAGNITUDEDIRECTION = 5
    M19DIRECTIONMAGNITUDE = 6
    M19CANCEL = 7
    M19LABEL = 8
"""
class ts_format(IntEnum):
    magnitude_direction = M19MAGNITUDEDIRECTION
    uv = M19REALREAL
    r_theta = M19MAGNITUDEDIRECTION


cdef Seconds temp
seconds = type(temp)
