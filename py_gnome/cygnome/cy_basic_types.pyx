"""
cython file used to store all the type info for GNOME.

Pulled from type_defs.pxi -- i.e pulled from C++ headers, etc.

"""

import cython

## pull stuff in from the C++ headers
from type_defs cimport *

# accessible in python
cdef public enum type_defs:
    status_not_released = OILSTAT_NOTRELEASED
    status_in_water = OILSTAT_INWATER
    status_on_land = OILSTAT_ONLAND
    status_off_maps = OILSTAT_OFFMAPS
    status_evaporated = OILSTAT_EVAPORATED
    disp_status_dont_disperse = DONT_DISPERSE
    disp_status_disperse = DISPERSE
    disp_status_have_dispersed = HAVE_DISPERSED
    disp_status_disperse_nat = DISPERSE_NAT
    disp_status_have_dispersed_nat = HAVE_DISPERSED_NAT
    disp_status_evaporate = EVAPORATE
    disp_status_have_evaporated = HAVE_EVAPORATED
    disp_status_remove = REMOVE
    disp_status_have_removed = HAVE_REMOVED

