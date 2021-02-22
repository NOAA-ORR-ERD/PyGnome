cimport numpy as cnp
import numpy as np

from .current_movers cimport CurrentCycleMover_c
from gnome.cy_gnome.cy_mover cimport CyMover


cdef class CyCurrentCycleMover(CyMover):
    cdef CurrentCycleMover_c *current_cycle
    cdef char *_num_method
