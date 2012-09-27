import cython
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from gnome import basic_types

from type_defs cimport * 
from mem_utils cimport _NewHandle, _GetHandleSize
from ossm_time cimport OSSMTimeValue_c

cdef class Cy_ossm_time:

    # underlying C++ object that is instantiated
    cdef OSSMTimeValue_c * time_dep

    def __cinit__(self):
       """ TODO: Update it so it can take path as input argument"""
       self.time_dep = new OSSMTimeValue_c()
        
    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self, path=None, timeseries=None):
        """
        Initialize object - takes either path or time value pair to initialize
        :param path: path to file containing time series data
        :param timeseries: numpy array containing time series data in time_value_pair structure as defined in type_defs
        If both are given, it will use the first keyword it finds
        """
#        if kwargs.items() == []:
#            raise IOError   # user didn't provide required input arguments
#        
#        match = False
#        for key in kwargs:
#            if key == 'path':
#                match = True
#                self.filename = kwargs.get('path')
#                
#                break
#                pass
#            elif key == 'timeseries':
#                match = True
#                break
#                pass
#            
#        if match == False:
#            raise IOError   # must provide at least one input
                
        pass
    
    def GetTimeValue(self, modelTime):
        """
          GetTimeValue - for a specified modelTime, gets the value
          CURRENTLY NOT WORKING 
        """
        cdef cnp.ndarray[Seconds, ndim=1] modelTimeArray
        modelTimeArray = np.asarray(modelTime, basic_types.seconds).reshape((-1,))     
         
        # velocity record passed to OSSMTimeValue_c methods and returned back to python
        cdef cnp.ndarray[VelocityRec, ndim=1] vel_rec 
        cdef VelocityRec * velrec
        
        cdef unsigned int i 
        cdef OSErr err 
        
        vel_rec = np.empty((modelTimeArray.size,), dtype=basic_types.velocity_rec)
        
        for i in range( 0, modelTimeArray.size):
            err = self.time_dep.GetTimeValue( modelTimeArray[i], &vel_rec[i])
            if err != 0:
                raise ValueError
            
        return vel_rec
    
       
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
        if err == 0:
            return err
        else:
            # TODO: raise an exception if err != 0
            raise IOError
    
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