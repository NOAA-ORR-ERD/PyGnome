cimport cython
from libc cimport stdlib
from libc.stdio cimport FILE, fopen, fread

import locale
import os
import sys

cimport numpy as cnp
import numpy as np

from gnome import basic_types
from .type_defs cimport Seconds, DateTimeRec
cimport gnome.cy_gnome.utils as utils

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


# no longer used -- filename_as_bytes is the one to use.
# cdef bytes to_bytes(unicode ucode):
#     """
#     Encode a string to its unicode type to default file system encoding for
#     the OS.

#     This is only a thin wrapper around os.fsencode().

#     The only point is for it to be a cdef function that explicitly takes
#     a unicode object -- which is to say, a py3 string.

#     We could probably deprecate this and just use os.fsencode() directly.

#     os.fsencode takes any PathLike object (e.g. pathlib.Path)

#     For the mac and most linuxes, it encodes it as utf-8.
# cdef bytes to_bytes(unicode ucode):
#     """
#     Encode a string to its unicode type to default file system encoding for
#     the OS.

#     This is only a thin wrapper around os.fsencode().

#     The only point is for it to be a cdef function that explicitly takes
#     a unicode object -- which is to say, a py3 string.

#     We could probably deprecate this and just use os.fsencode() directly.

#     os.fsencode takes any PathLike object (e.g. pathlib.Path)

#     For the mac and most linuxes, it encodes it as utf-8.

#     For Windows this is usually cp1252, though could be different.

#     This should work for any C/C++ code that uses a char * for filenames.
#     """
#     cdef bytes byte_string

#     byte_string = os.fsencode(ucode)

#     For Windows this is usually cp1252, though could be different.

#     This should work for any C/C++ code that uses a char * for filenames.
#     """
#     cdef bytes byte_string

#     byte_string = os.fsencode(ucode)

#     return byte_string

cpdef bytes filename_as_bytes(filepath):
    '''
    filename is any python "pathlike" object

    usually a string or pathlib.Path object

    returns a bytes object, in the local filesystem encoding

    NOTE: this is pretty much just a wrapper around os.fsencode()
    (it also calls os.path.normpath)

    This *should* work on windows -- but doesn't, so restricting Windows to latin-1
    '''

    cdef bytes file_
    filepath = os.path.normpath(filepath)
    # If the file doesn't exist, it doesn't matter if it's encoded properly
    #   and we want users to get the FileNotFound Error
    #if os.path.exists(filepath):
        # This should get removed if we can figure out how to do this right
        # Windows *should* support all of Unicode
    if sys.platform.startswith("win"):
        try:
            file_ = filepath.encode('cp1252')
            return file_
        except UnicodeEncodeError:
            raise ValueError("gnome only supports latin filenames (cp-1252) on this system:\n"
                             f"{filepath} not supported")
    else:
        try:
            file_ = os.fsencode(filepath)
        except UnicodeEncodeError:
            raise ValueError("Filename: {} is not legal on this system".format(filepath))

    return file_

def read_file(path):
    """
    reads the ascii text from a file

    only here to test the filename_as_bytes function
    """

    cdef bytes bpath = filename_as_bytes(path)

    cdef FILE* fp
    fp = fopen(bpath, "r") # The type of "p" is "FILE*", as returned by fopen().

    if fp is NULL:
        raise FileNotFoundError("No such file or directory: " + str(path))
    # else:
    #     print("file opened:", bpath)

    cdef char buffer[101]

    num_read = fread(buffer, 1, 100, fp)

    # contents = bytearray(100)
    # for i in range(100):
    #      contents[i] =  buffer[i]
    cdef bytes contents = buffer #[:100]

    # return buffer[:100]
    return contents[:num_read]


