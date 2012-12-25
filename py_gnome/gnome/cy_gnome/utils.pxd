"""
lib_gnome utils
"""
from type_defs cimport *    
from libcpp cimport bool
from libcpp.string cimport string

"""
MemUtils functions available from lib_gnome 
"""
cdef extern from "MemUtils.h":
    Handle _NewHandle(long )
    void _DisposeHandleReally(Handle)
    long _GetHandleSize(Handle)

"""
Expose DateTime conversion functions from the lib_gnome/StringFunctions.h
"""
cdef extern from "StringFunctions.h":
    void DateToSeconds (DateTimeRec *, unsigned long *)
    void SecondsToDate (unsigned long, DateTimeRec *)

"""
Declare methods for interpolation of timeseries from 
lib_gnome/OSSMTimeValue_c class and ShioTimeValue
"""
cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        OSSMTimeValue_c() except +    
        OSErr   GetTimeValue(Seconds &, VelocityRec *)
        OSErr   ReadTimeValues (char *, short, short)
        void    SetTimeValueHandle(TimeValuePairH)    # sets all time values 
        TimeValuePairH GetTimeValueHandle()
        short   GetUserUnits()
        void    SetUserUnits(short)
        void    Dispose()
        
"""
ShioTimeValue_c.h derives from OSSMTimeValue_c - so no need to redefine methods given in OSSMTimeValue_c like GetTimeValue
"""
cdef extern from "ShioTimeValue_c.h":
        
    # TODO: These may not be needed
    ctypedef struct EbbFloodData:
        Seconds time
        double speedInKnots
        short type
        
    ctypedef EbbFloodData *EbbFloodDataP    # Weird syntax, it says EbbFloodDataP is pointer to pointer to EbbFloodData struct
    ctypedef EbbFloodData **EbbFloodDataH   # Weird syntax, it says EbbFloodDataH is pointer to pointer to EbbFloodData struct

    ctypedef struct HighLowData:
        Seconds time
        double height
        short type
    
    ctypedef HighLowData *HighLowDataP
    ctypedef HighLowData **HighLowDataH
    #==================
    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c() except +
        string      fStationName    # make char array a string - easier to work with in Cython
        char        fStationType
        bool        daylight_savings_off    # is this required?
        EbbFloodDataH   fEbbFloodDataHdl    # values to show on list for tidal currents - not sure if these should be available
        HighLowDataH    fHighLowDataHdl
        
        OSErr       ReadTimeValues (char *path)
        WorldPoint  GetRefWorldPoint()
        
        # Not Sure if Following are required/used
        OSErr       GetConvertedHeightValue(Seconds  , VelocityRec *)
        OSErr       GetProgressiveWaveValue(Seconds &, VelocityRec *)
