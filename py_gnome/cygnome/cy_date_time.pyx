import cython
cimport numpy as np
from gnome import basic_types

from date_time cimport * 

cdef class Cy_date_time:
   cdef unsigned long * seconds
   cdef unsigned long tSeconds
   cdef DateTimeRec * dateRec
   cdef DateTimeRec tDateRec
   
   def __cinit__(self):
       self.seconds = &self.tSeconds
       self.dateRec = &self.tDateRec
        
   def __dealloc__(self):
       """
           No python or cython objects need to be deallocated.
           Tried deleting seconds, tSeconds, dateVal here, but compiler
           throws errors
       """
       pass
       
   def DateToSeconds(self, np.ndarray[DateTimeRec, ndim=1] date):
       DateToSeconds( &date[0], self.seconds)
       return self.tSeconds

   def SecondsToDate(self, unsigned long secs):
       SecondsToDate( secs, self.dateRec)
       return self.tDateRec
