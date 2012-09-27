"""
Expose functions from the lib_gnome/StringFunctions.h
"""

from type_defs cimport DateTimeRec

cdef extern from "StringFunctions.h":
    void DateToSeconds (DateTimeRec *, unsigned long *)
    void SecondsToDate (unsigned long, DateTimeRec *)
