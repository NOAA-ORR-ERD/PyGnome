from libc cimport stdlib
import locale

cimport numpy as cnp
import numpy as np

from gnome import basic_types
from type_defs cimport Seconds, DateTimeRec
cimport utils

cdef class CyDateTime:

    def __dealloc__(self):
        """
            No python or cython objects need to be deallocated.
            Tried deleting seconds, tSeconds, dateVal here, but compiler
            throws errors
        """
        pass

    def DateToSeconds(self, cnp.ndarray[DateTimeRec, ndim=1] date):
        cdef Seconds seconds

        utils.DateToSeconds(&date[0], &seconds)

        return seconds

    def SecondsToDate(self, Seconds secs):
        cdef cnp.ndarray[DateTimeRec, ndim = 1] daterec

        daterec = np.empty((1, ), dtype=basic_types.date_rec)
        utils.SecondsToDate(secs, &daterec[0])

        return daterec[:][0]


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
    Encode a string to its unicode type to default file system encoding for
    the OS.
    It uses locale.getpreferredencoding() to get the filesystem encoding
        For the mac it encodes it as utf-8.
        For windows this appears to be cp1252.

    The C++ expects char * so  either of these encodings appear to work. If the
    getpreferredencoding returns a type of encoding that is incompatible with
    a char * C++ input, then things will fail.
    """
    cdef bytes byte_string

    try:
        byte_string = ucode.encode(locale.getpreferredencoding())
    except Exception as err:
        raise err

    return byte_string
