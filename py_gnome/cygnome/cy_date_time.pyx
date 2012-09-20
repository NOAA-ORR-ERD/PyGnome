"""
Cython wrapper around Gnome C++ DateToSeconds and SecondsToDate
functions defined in "StringFuncitons.h"

NOTE: 
    Both Gnome C++ and Python use the standard C mktime function
    in DateToSeconds. The tm_isdst = 0 so daylight savings is not 
    taken into account for the conversion
"""

import cython
cimport numpy as np

include "type_def.pxi"
include "date_time.pxi"

cdef class Cy_string_functions:
   cdef unsigned long * seconds
   cdef unsigned long tSeconds
   
   def __cinit__(self):
       self.seconds = &self.tSeconds
        
   def __dealloc__(self):
       """
           No python or cython objects need to be deallocated.
           Tried deleting seconds, tSeconds, dateVal here, but compiler
           throws errors
       """
       pass
       
   def DateToSeconds(self, np.ndarray[DateTimeRec, ndim=1] date):
       """Converts Date to time in seconds since Jan 1st 1970 - dst flag = 0"""
       DateToSeconds( &date[0], self.seconds)
       return self.tSeconds
   
   def SecondsToDate(self, secs):
       daterec = np.empty((1,), dtype=basic_types.date_rec)
       self.tSeconds = secs
       SecondsToDate( self.seconds, &daterec[0])
       return daterec