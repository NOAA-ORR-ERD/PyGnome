import cython
import numpy as np
import os 

cimport numpy as cnp
from libc.string cimport memcpy

from gnome import basic_types

from type_defs cimport * 
from mem_utils cimport _NewHandle, _GetHandleSize
from ossm_time cimport OSSMTimeValue_c

cdef class Cy_ossm_time:

    # underlying C++ object that is instantiated
    cdef OSSMTimeValue_c * time_dep
    
    # PYTHON CANNOT ACCESS THESE ATTRIBUTES DIRECTLY
    # class attributes so that once the object is properly initialized
    # we don't need to do any memcpy to get the time series
    #     Cy_ossm_time.timeseries gives the time series
    #     Cy_ossm_time.untis gives the units
    #cdef cnp.ndarray[TimeValuePair, ndim=1] timeseries    # Cannot do this!
    cdef cnp.ndarray tSeries

    def __cinit__(self):
       """ TODO: Update it so it can take path as input argument"""
       self.time_dep = new OSSMTimeValue_c(NULL)
        
    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self, path=None, file_contains=None, cnp.ndarray[TimeValuePair, ndim=1] timeseries=None, units=-1):
        """
        Initialize object - takes either path or time value pair to initialize
        :param path: path to file containing time series data
        :param timeseries: numpy array containing time series data in time_value_pair structure as defined in type_defs
        If both are given, it will use the first keyword it finds
        """
        if path is None and timeseries is None:
            raise ValueError("Object needs either a path to the time series file or timeseries")   # TODO: check error
        
        if path is not None:
            if file_contains is None:
                raise ValueError('Unknown file contents - need a valid basic_types.file_contains.* value')
            
            if os.path.exists(path):
                self._ReadTimeValues(path, file_contains, units)
            else:
                raise IOError("No such file: " + path)
        else:
            # units must not be -1 in this case
            if units == -1:
                raise ValueError('Valid units must be provided with timeseries')
            
            if timeseries is None:
                raise ValueError("timeseries cannot be None")
            
            self._SetTimeValueHandle(timeseries)
            self.time_dep.SetUserUnits(units)
            
            self.tSeries = self._GetTimeValueHandle()    # make sure it is set correctly for OSSMTime Object
            
    
    @property
    def Units(self):
        """returns the class attribute for units - couldn't see the class attribute in python so had to return it
        as part of the method"""
        return self.time_dep.GetUserUnits()
    
    @property
    def Timeseries(self):
        """returns the class attribute for time series - couldn't see the class attribute in python so had to return it
        as part of the method. This is just so we only do a memcpy once"""
        return self.tSeries
    
    def GetTimeValue(self, modelTime):
        """
          GetTimeValue - for a specified modelTime or array of model times, it returns the values
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
    
       
    def _ReadTimeValues(self, path, file_contains, units=-1):
        """
            Format for the data file. This is an enum type in C++
            defined below. These are defined in cy_basic_types such that python can see them
            
            #===========================================================================
            # enum { M19REALREAL = 1, M19HILITEDEFAULT, M19MAGNITUDEDEGREES, M19DEGREESMAGNITUDE,
            # M19MAGNITUDEDIRECTION, M19DIRECTIONMAGNITUDE,M19CANCEL, M19LABEL };
            #===========================================================================
            
            The default format is Magnitude and direction as defined for wind
            
            Units are defined by following integers:
                Knots: 1
                MilesPerHour: 2
                MetersPerSec: 3
                
            Make this private since the constructor will likely call this when object is instantiated
        """        
        err = self.time_dep.ReadTimeValues(path, file_contains, units)
        if err == 0:
            # let's set class attributes
            self.tSeries = self._GetTimeValueHandle()
            
        elif err == 1:
            # TODO: need to define error codes in C++ and raise other exceptions
            raise ValueError("Units not found in file and units not provided as input")
    
    def _SetTimeValueHandle(self, cnp.ndarray[TimeValuePair, ndim=1] time_val):
        """
            Takes a numpy array containing a time series, copies it to a Handle (TimeValuePairH),
        then invokes the SetTimeValueHandle method of OSSMTimeValue_c object
        Make this private since the constructor will likely call this when object is instantiated
        """
        if time_val is None:
            raise TypeError("expected ndarray, NoneType found")
        
        cdef short tmp_size = sizeof(TimeValuePair)
        cdef TimeValuePairH time_val_hdlH
        time_val_hdlH = <TimeValuePairH>_NewHandle(time_val.nbytes)
        memcpy( time_val_hdlH[0], &time_val[0], time_val.nbytes)
        self.time_dep.SetTimeValueHandle(time_val_hdlH)
    
    def _GetTimeValueHandle(self): 
        """
            Invokes the GetTimeValueHandle method of OSSMTimeValue_c object to read the time series data
        """
        cdef short tmp_size = sizeof(TimeValuePair)
        cdef TimeValuePairH time_val_hdlH
        cdef cnp.ndarray[TimeValuePair, ndim=1] tval
        
        time_val_hdlH = self.time_dep.GetTimeValueHandle()
        sz = _GetHandleSize(<Handle>time_val_hdlH)  # allocate memory and copy it over
        tval = np.empty((sz/tmp_size,), dtype=basic_types.time_value_pair)  # will this always work?
        
        memcpy( &tval[0], time_val_hdlH[0], sz)
        return tval