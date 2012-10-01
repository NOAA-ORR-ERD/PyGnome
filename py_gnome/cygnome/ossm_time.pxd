"""
Declare methods from lib_gnome/OSSMTimeValue_c class
"""

from type_defs cimport *
from mover cimport Mover_c

cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        OSSMTimeValue_c(Mover_c *) except +
        #OSSMTimeValue_c() except +    # not sure why it never enters an empty constructor!
        OSErr GetTimeValue(Seconds &, VelocityRec *)
        OSErr ReadTimeValues (char *, short, short)
        void SetTimeValueHandle(TimeValuePairH)	# sets all time values 
        TimeValuePairH GetTimeValueHandle()
        short GetUserUnits()
        void  SetUserUnits(short)