import cython
cimport numpy as np
from libc.string cimport memcpy

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
       err = self.time_dep.ReadTimeValues(path, format, units)
       return err
        
    #===========================================================================
    # def SetTimeValueHandle(self, np.ndarray[TimeValuePair, ndim=1] time_val):
    #   #cdef TimeValuePairP time_val_hdlP
    #   #time_val_hdlP = &time_val[0]
    #   
    #   cdef TimeValuePairH time_val_hdlH
    #   #time_val_hdlH = &time_val_hdlP
    #   
    #   cdef short tmp_size = sizeof(TimeValuePair)
    #   memcpy(time_val_hdlH, &time_val[0], time_val.size*tmp_size)
    #   
    #   self.time_dep.SetTimeValueHandle(time_val_hdlH)
    #   pass
    #===========================================================================