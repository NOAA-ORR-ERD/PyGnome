import numpy as np
import os 

cimport numpy as cnp
from libc.string cimport memcpy

from gnome import basic_types

from type_defs cimport * 
from utils cimport _NewHandle, _GetHandleSize
from utils cimport OSSMTimeValue_c

from gnome.utilities.convert import to_bytes

cdef class CyOSSMTime(object):

    # underlying C++ object that is instantiated
    # declared in pxd file
    #cdef OSSMTimeValue_c * time_dep
    

    def __cinit__(self):
        self.time_dep = new OSSMTimeValue_c()
        
    def __dealloc__(self):
        del self.time_dep
    
    def __init__(self, filename=None, file_contains=None, cnp.ndarray[TimeValuePair, ndim=1] timeseries=None, scale_factor=1):
        """
        Initialize object - takes either file or time value pair to initialize
        :param file: path to file containing time series data. It valid user_units are defined in the file, it uses them; otherwise,
        it defaults the user_units to meters_per_sec.
        :param file_contains: enum type defined in cy_basic_types to define contents of data file
        :param timeseries: numpy array containing time series data in time_value_pair structure as defined in type_defs
        If both are given, it will read data from the file
        
        NOTE: If timeseries are given, and data is velocity, it is always assumed to be in meters_per_sec
        """
        if filename is None and timeseries is None:
            raise ValueError("Object needs either a file to the time series file or timeseries")   # TODO: check error
        
        if filename is not None:
            if file_contains is None:
                raise ValueError('Unknown file contents - need a valid basic_types.ts_format.* value')
            
            # check file_contains is valid parameter - both magnitude_direction and uv are valid inputs
            if file_contains not in basic_types.ts_format._int:
                raise ValueError("file_contains can only contain integers 5, or 1; also defined by basic_types.ts_format.<magnitude_direction or uv>")
            
            if os.path.exists(filename):
                self._read_time_values(filename, file_contains, -1) # user_units should be read from the file
            else:
                raise IOError("No such file: " + filename)
        
        elif timeseries is not None:
            self._set_time_value_handle(timeseries)
            self.time_dep.SetUserUnits(-1)  # UserUnits for velocity assumed to be meter per second. 
                                            # Leave undefined because the timeseries could be something other than velocity
                                            # TODO: check if OSSMTimeValue_c is only used for velocity data?
        self.time_dep.fScaleFactor = scale_factor
    property user_units:
        def __get__(self):
            """
            returns units for the time series
            Define units for velocity. In C++, these are #defined as
            #define kUndefined      -1
            #define kKnots           1
            #define kMetersPerSec    2
            #define kMilesPerHour    3
            """
            if self.time_dep.GetUserUnits() == -1:
                return "undefined"
            elif self.time_dep.GetUserUnits() == 1:
                return "knot"
            elif self.time_dep.GetUserUnits() == 2:
                return "meter per second"
            elif self.time_dep.GetUserUnits() == 3:
                return "mile per hour"
            else:
                raise ValueError("C++ GetUserUnits() gave a result which is outside the expected bounds.")
    
    property timeseries:
        def __get__(self):
            """returns the time series stored in the OSSMTimeValue_c object. It returns a memcpy of it."""
            return self._get_time_value_handle()
        
        def __set__(self, value):
            self._set_time_value_handle(value)
    
    property filename:
        def __get__(self):
            return <bytes>self.time_dep.fileName

    property scale_factor:
        def __get__(self):
            return self.time_dep.fScaleFactor
        
        def __set__(self,value):
            self.time_dep.fScaleFactor = value        


    property station_location:
        def __get__(self):
            return np.array((0,0,0), dtype=basic_types.world_point)    # will replace this once OSSMTime contains values
        
        def __set__(self, value):
            self.station_location = value
    
    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be recreated 
        """
        # Tried the following, but eval(repr( obj_instance)) would not work on it so updated it to hard code the class name
        # '{0.__class__}( "{0.filename}", daylight_savings_off={1})'.format(self, self.shio.daylight_savings_off)
        return 'CyOSSMTime( filename="{0.filename}", timeseries=<see timeseries attribute>)'.format(self)
      
    def __str__(self):
        """Return string info about the object"""
        info  = "CyOSSMTime object - filename={0.filename}, timeseries=<see timeseries attribute>".format(self)
        
        return info
    
    def get_time_value(self, modelTime):
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
                raise ValueError("Error invoking TimeValue_c.GetTimeValue method in CyOSSMTime: C++ OSERR = " + str(err))
            
        return vel_rec
    
       
    def _read_time_values(self, filename, file_contains, user_units=-1):
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
        cdef bytes file_
        file_= <bytes> to_bytes(filename)
        err = self.time_dep.ReadTimeValues( file_, file_contains, user_units)
        if err == 1:
            # TODO: need to define error codes in C++ and raise other exceptions
            raise ValueError("Valid user units not found in file")
    
    def _set_time_value_handle(self, cnp.ndarray[TimeValuePair, ndim=1] time_val):
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
    
    def _get_time_value_handle(self): 
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
