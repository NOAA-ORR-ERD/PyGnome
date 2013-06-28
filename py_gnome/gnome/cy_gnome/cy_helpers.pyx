from libc cimport stdlib
import locale

cimport numpy as np

from gnome import basic_types
cimport type_defs
cimport utils

cdef class CyDateTime:
    cdef type_defs.Seconds * seconds
    cdef type_defs.Seconds tSeconds
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
    
    def SecondsToDate(self, type_defs.Seconds secs):
        utils.SecondsToDate( secs, self.dateRec)
        return self.tDateRec


def srand(seed):
    """
    Resets C random seed 
    """
    stdlib.srand(seed)
    
def rand():
    """
    Calls the C stdlib.rand() function
    
    Only implemented for testing that the srand was set correctly
    """
    return stdlib.rand()


cdef bytes to_bytes(unicode ucode):
    """
    Encode a string to its unicode type to default file system encoding for the OS
    It uses locale.getpreferredencoding() to get the filesystem encoding
        For the mac it encodes it as utf-8.
        For windows this appears to be cp1252.
        
    The C++ expects char * so  either of these encodings appear to work. If the 
    getpreferredencoding returns a type of encoding that is incompatible with a char *
    C++ input, then things will fail.
    """
    cdef bytes byte_string
    
    try:
        byte_string = ucode.encode(locale.getpreferredencoding())
    except Exception as err:
        raise err
    
    return byte_string
