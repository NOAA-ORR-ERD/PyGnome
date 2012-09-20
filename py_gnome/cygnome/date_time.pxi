"""
Expose functions from the lib_gnome/StringFunctions.h
"""

include "type_defs.pxi"

cdef extern from "StringFunctions.h":
    void DateToSeconds (DateTimeRec *, unsigned long *)
    void SecondsToDate (unsigned long, DateTimeRec *)
