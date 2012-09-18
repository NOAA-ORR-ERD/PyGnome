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

#    def __cinit__(self, np.ndarray[TimeValuePair, ndim=1] time_vals, units=1):
# 
#        """
#        units:
# 
#        kCMS = 1
#        kKCMS = 2
#        kCFS = 3
#        kKCFS = 3
# 
#        (If you can make sense of any of this.)
# 
#	    """
#        #self.time_dep = new OSSMTimeValue_c(NULL, &time_vals[0], units)
#        cdef TimeValuePairH time_val_hdl
#        cdef short tmp_size = sizeof(TimeValuePair)
#        memcpy(time_val_hdl, <char *>deref(time_vals.data), time_vals.size*tmp_size)
#        self.time_dep = new OSSMTimeValue_c(NULL, time_val_hdl, units)
        
        
    def __dealloc__(self):
        del self.time_dep
    
    #def __init__(self, np.ndarray[TimeValuePair] time_vals, units=1):
    def __init__(self):
        """
        Initialize object
        """
        pass
    
    def GetTimeValue(self, modelTime, np.ndarray[VelocityRec, ndim=1] value):
      """
      GetTimeValue - for a specified modelTime, gets the value 
      """
      err = self.time_dep.GetTimeValue( modelTime, &value[0])
       
    def ReadTimeValues(self, path, format=5, units=1):
       #===========================================================================
       # enum { M19REALREAL = 1, M19HILITEDEFAULT,
       # 
       # M19MAGNITUDEDEGREES, M19DEGREESMAGNITUDE,
       # 
       # M19MAGNITUDEDIRECTION, M19DIRECTIONMAGNITUDE,
       # 
       # M19CANCEL, M19LABEL };
       #===========================================================================
       err = self.time_dep.ReadTimeValues(path, 5, 1)
       return err
        
#    def SetTimeValueHandle(self, np.ndarray[TimeValuePair, ndim=1] time_vals):
#       cdef TimeValuePairH time_val_hdl
#       #cdef short tmp_size = sizeof(TimeValuePair)
#       #memcpy(time_val_hdl, <char *>deref(time_vals.data), time_vals.size*tmp_size)
#       time_val_hdl = &time_vals[0]
#       pass
        