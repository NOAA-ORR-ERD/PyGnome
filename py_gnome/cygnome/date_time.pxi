"""
Expose DatTime functions from the lib_gnome/StringFunctions.h
"""

cdef extern from "StringFunctions.h":
	void DateToSeconds(DateTimeRec *, unsigned long *);
	void SecondsToDate(unsigned long *, DateTimeRec *);