import cython
cimport numpy as np
from libc.string cimport memcpy
cimport libc.stdlib as stdlib
cimport libc.stdio as stdio
from cython.operator cimport dereference as deref

include "ossm_time.pxi"

cdef class Cy_ossm_time:

    cdef OSSMTimeValue_c *time_dep

    def __cinit__(self):
       self.time_dep = new OSSMTimeValue_c(NULL)

    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self):
        """
        Initialize object
        """
        pass
    
    def GetTimeValue(self, modelTime, np.ndarray[VelocityRec, ndim=1] value):
      """
      GetTimeValue - for a specified modelTime, it interpolates and returns the value 
      """
      err = self.time_dep.GetTimeValue( modelTime, &value[0])
       
    def ReadTimeValues(self, path, format=5, units=1):
        """
        Read the TimeValues from a datafile. Currently, the format defaults to the
        format of the Wind Data File. The units default to kKnots.
        
        Format is an enum type, defined as follows:
        #===========================================================================
        # enum { M19REALREAL = 1, M19HILITEDEFAULT,M19MAGNITUDEDEGREES, M19DEGREESMAGNITUDE,
        #        M19MAGNITUDEDIRECTION, M19DIRECTIONMAGNITUDE,M19CANCEL, M19LABEL };
        #===========================================================================

        Similarly, units are defined by an integer as follows:
        # kKnots            1
        # kMetersPerSec    2
        # kMilesPerHour    3
        """
        err = self.time_dep.ReadTimeValues(path, format, units)
        return err
        
        