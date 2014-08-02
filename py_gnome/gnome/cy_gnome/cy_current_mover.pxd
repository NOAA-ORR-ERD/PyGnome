'Base class for all current movers'

from movers cimport Mover_c
from current_movers cimport CurrentMover_c, CATSMover_c
from cy_mover cimport CyMover

cdef extern from *:
    CurrentMover_c* dc_mover_to_cmover "dynamic_cast<CurrentMover_c *>" \
        (Mover_c *) except NULL
    CATSMover_c* dc_cmover_to_cats "dynamic_cast<CATSMover_c *>" \
        (CurrentMover_c *) except NULL

cdef class CyCurrentMover(CyMover):
    cdef CurrentMover_c * curr_mover
