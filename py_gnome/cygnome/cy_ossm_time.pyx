import cython
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from gnome import basic_types

include "ossm_time.pxi"
include "mem_utils.pxi"

cdef class Cy_ossm_time:

    # underlying C++ object that is instantiated
    cdef OSSMTimeValue_c * time_dep
    
    # velocity record passed to OSSMTimeValue_c methods and returned back to python
    cdef VelocityRec * velrec
    cdef VelocityRec tVelRec
    

    def __cinit__(self):
       self.time_dep = new OSSMTimeValue_c(NULL)
       self.velrec = &self.tVelRec
        
    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self):
        """
            Initialize object
        """
        pass
    
    def GetTimeValue(self, modelTime):
        """
          GetTimeValue - for a specified modelTime, gets the value 
        """
        err = self.time_dep.GetTimeValue( modelTime, self.velrec)
        if err == 0:
            return self.tVelRec
        else:
            # TODO: raise an exception if err != 0
            raise IOError
          
       
    def ReadTimeValues(self, path, format=5, units=1):
        """
            Format for the data file. This is an enum type in C++
            defined below. 
            
            #===========================================================================
            # enum { M19REALREAL = 1, M19HILITEDEFAULT, M19MAGNITUDEDEGREES, M19DEGREESMAGNITUDE,
            # M19MAGNITUDEDIRECTION, M19DIRECTIONMAGNITUDE,M19CANCEL, M19LABEL };
            #===========================================================================
            
            The default format is Magnitude and direction as defined for wind
            
            Units are defined by following integers:
                Knots: 1
                MilesPerHour: 2
                MetersPerSec: 3
        """
        err = self.time_dep.ReadTimeValues(path, format, units)
        return err
    
    def SetTimeValueHandle(self, cnp.ndarray[TimeValuePair, ndim=1] time_val):
        """Takes a numpy array containing a time series, copies it to a Handle (TimeValuePairH),
        then invokes the SetTimeValueHandle method of OSSMTimeValue_c object"""
        cdef short tmp_size = sizeof(TimeValuePair)
        cdef TimeValuePairH time_val_hdlH
        time_val_hdlH = <TimeValuePairH>_NewHandle(time_val.nbytes)
        memcpy( time_val_hdlH[0], &time_val[0], time_val.nbytes)   
        self.time_dep.SetTimeValueHandle(time_val_hdlH)
    
    def GetTimeValueHandle(self): 
        """Invokes the GetTimeValueHandle method of OSSMTimeValue_c object to read the time series data"""
        cdef short tmp_size = sizeof(TimeValuePair)
        cdef TimeValuePairH time_val_hdlH
        cdef cnp.ndarray[TimeValuePair, ndim=1] tval
        
        time_val_hdlH = self.time_dep.GetTimeValueHandle()
        sz = _GetHandleSize(<Handle>time_val_hdlH)  # allocate memory and copy it over
        tval = np.empty((sz/tmp_size,), dtype=basic_types.time_value_pair)  # will this always work?
        
        memcpy( &tval[0], time_val_hdlH[0], sz)
        return tval