cimport numpy as cnp
import numpy as np
from current_movers cimport GridCurrentMover_c
from gnome.cy_gnome.cy_mover cimport CyCurrentMoverBase


cdef class CyGridCurrentMover(CyCurrentMoverBase):
    cdef GridCurrentMover_c *grid_current