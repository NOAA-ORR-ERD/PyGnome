import cython
cimport numpy as np
from libc.string cimport memcpy
cimport libc.stdlib as stdlib
cimport libc.stdio as stdio
from cython.operator cimport dereference as deref
import numpy as nmp

include "ossm_time.pxi"

cdef class ossm_time:

    cdef OSSMTimeValue_c *time_dep

    def __cinit__(self, np.ndarray[TimeValuePair] time_vals, units=1):

        """
        units:

        kKnots = 1
        kMetersPerSec = 2
        kMilesPerHour = 3

	"""

        cdef TimeValuePairH time_val_hdl
        cdef short tmp_size = sizeof(TimeValuePair)
        memcpy(time_val_hdl, <char *>deref(time_vals.data), time_vals.size*tmp_size)
        self.time_dep = new OSSMTimeValue_c(NULL, time_val_hdl, units)
        
    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self, np.ndarray[TimeValuePair] time_vals, units=1):
        pass

