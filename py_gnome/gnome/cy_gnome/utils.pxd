"""
lib_gnome utils
"""
from type_defs cimport *
from libcpp cimport bool

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
        OSSMTimeValue_c() except +    # not sure why it never enters an empty constructor!
        OSErr GetTimeValue(Seconds &, VelocityRec *)
        OSErr ReadTimeValues (char *, short, short)
        void SetTimeValueHandle(TimeValuePairH)    # sets all time values 
        TimeValuePairH GetTimeValueHandle()
        short GetUserUnits()
        void  SetUserUnits(short)
        void Dispose()
        
cdef extern from "ShioTimeValue_c.h":
    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c()
        OSErr    ReadTimeValues (char *path)
        WorldPoint GetRefWorldPoint()
        
