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

"""
LE Status as an enum type
"""
oil_status =enum(not_released = OILSTAT_NOTRELEASED,
                 in_water = OILSTAT_INWATER,
                 on_land = OILSTAT_ONLAND,
                 off_maps = OILSTAT_OFFMAPS,
                 evaporated = OILSTAT_EVAPORATED)

"""
disperse status as an enum type
"""
disp_status = enum(dont_disperse = DONT_DISPERSE,
                   disperse = DISPERSE,
                   have_dispersed = HAVE_DISPERSED,
                   disperse_nat = DISPERSE_NAT,
                   have_dispersed_nat = HAVE_DISPERSED_NAT,
                   evaporate = EVAPORATE,
                   have_evaporated = HAVE_EVAPORATED,
                   remove = REMOVE,
                   have_removed = HAVE_REMOVED)

"""
SpillType {FORECAST_LE = 1, UNCERTAINTY_LE = 2};
"""
spill_type = enum(forecast = FORECAST_LE,
                  uncertainty = UNCERTAINTY_LE,)
                

"""
Contains enum type for the contents of a data file. For instance,
a standard wind file would contain magnitude and direction info
data_format.magnitude_direction = 5
"""
data_format = enum(magnitude_direction=M19MAGNITUDEDIRECTION,
                   wind_uv=M19REALREAL)

"""
Lets define error codes here as well, may want to 
group these under an ErrCodes class if it makes sense - but for now, let's see
how many error codes we get.
"""
err_codes = enum(undefined_units=1)
#class ErrCodes:
#    undefined = enum(units=1)
