"""
cython file used to store all the type info for GNOME.

Pulled from type_defs.pxi -- i.e pulled from C++ headers, etc.

"""

import cython

## pull stuff in from the C++ headers
from type_defs cimport *

def enum(**enums):
    """
    Just found a clever way to do enums in python
    """
    return type('Enum', (), enums)

## accessible in python 
## TODO: Delete this after replacing with le_status and disp_status
#cdef public enum type_defs:
#    status_not_released = OILSTAT_NOTRELEASED
#    status_in_water = OILSTAT_INWATER
#    status_on_land = OILSTAT_ONLAND
#    status_off_maps = OILSTAT_OFFMAPS
#    status_evaporated = OILSTAT_EVAPORATED
#    disp_status_dont_disperse = DONT_DISPERSE
#    disp_status_disperse = DISPERSE
#    disp_status_have_dispersed = HAVE_DISPERSED
#    disp_status_disperse_nat = DISPERSE_NAT
#    disp_status_have_dispersed_nat = HAVE_DISPERSED_NAT
#    disp_status_evaporate = EVAPORATE
#    disp_status_have_evaporated = HAVE_EVAPORATED
#    disp_status_remove = REMOVE
#    disp_status_have_removed = HAVE_REMOVED


"""
LE Status as an enum type
"""
oil_status =enum(status_not_released = OILSTAT_NOTRELEASED,
                 status_in_water = OILSTAT_INWATER,
                 status_on_land = OILSTAT_ONLAND,
                 status_off_maps = OILSTAT_OFFMAPS,
                 status_evaporated = OILSTAT_EVAPORATED)

"""
disperse status as an enum type
"""
disp_status = enum(disp_status_dont_disperse = DONT_DISPERSE,
                disp_status_disperse = DISPERSE,
                disp_status_have_dispersed = HAVE_DISPERSED,
                disp_status_disperse_nat = DISPERSE_NAT,
                disp_status_have_dispersed_nat = HAVE_DISPERSED_NAT,
                disp_status_evaporate = EVAPORATE,
                disp_status_have_evaporated = HAVE_EVAPORATED,
                disp_status_remove = REMOVE,
                disp_status_have_removed = HAVE_REMOVED)

"""
Contains enum type for the contents of a data file. For instance,
a standard wind file would contain magnitude and direction info
file_contais.magnitude_direction = 5
"""
file_contains = enum(magnitude_direction=M19MAGNITUDEDIRECTION)

"""
Define units for velocity. In C++, these are #defined as
#define kKnots           1
#define kMetersPerSec    2
#define kMilesPerHour    3
"""
velocity_units = enum(knots=1, meters_per_sec=2, miles_per_hour=3)

"""
Lets define error codes here as well, may want to 
group these under an ErrCodes class if it makes sense - but for now, let's see
how many error codes we get.
"""
err_codes = enum(undefined_units=1)
#class ErrCodes:
#    undefined = enum(units=1)
