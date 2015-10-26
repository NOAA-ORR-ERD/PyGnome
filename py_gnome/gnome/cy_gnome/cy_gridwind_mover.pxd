cimport numpy as cnp
import numpy as np
from movers cimport GridWindMover_c
from gnome.cy_gnome.cy_mover cimport CyWindMoverBase


cdef class CyGridWindMover(CyWindMoverBase):
    cdef GridWindMover_c *grid_wind