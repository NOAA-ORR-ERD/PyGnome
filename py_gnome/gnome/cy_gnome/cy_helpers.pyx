cimport numpy as np
from gnome import basic_types

cimport type_defs
cimport utils
cimport stdlib

cdef class CyDateTime:
    cdef unsigned long * seconds
    cdef unsigned long tSeconds
    cdef type_defs.DateTimeRec * dateRec
    cdef type_defs.DateTimeRec tDateRec
   
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
       
    def DateToSeconds(self, np.ndarray[type_defs.DateTimeRec, ndim=1] date):
        utils.DateToSeconds( &date[0], self.seconds)
        return self.tSeconds

    def SecondsToDate(self, unsigned long secs):
        utils.SecondsToDate( secs, self.dateRec)
        return self.tDateRec


def srand(seed):
    """
    Resets C++ random seed 
    """
    stdlib.srand(seed)
