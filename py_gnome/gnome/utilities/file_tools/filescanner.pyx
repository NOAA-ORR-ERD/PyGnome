"""
cython-optimized code for scanning text files and reading numbers out of them

based on the classic C fscanf
"""

import sys

cimport cython
import numpy as np
cimport numpy as cnp
cimport libc
from libc cimport stdio
from libc.stdint cimport uint32_t, UINT32_MAX
from cpython cimport *




## NOTE: this to get the file pointer form the python file object
##       does not work in Py3!

# cdef extern from "Python.h":
#     ## dont know why I need this -- shouldn't it be in the cpython pxd?
#     stdio.FILE* PyFile_AsFile(PyObject*)
#     int PyFile_CheckExact(PyObject*)

#    void  fprintf(FILE* f, char* s, char* s)
# Next, enter the builtin file class into the namespace:
#cdef extern from "fileobject.h":
#    ctypedef class __builtin__.file [object PyFileObject]:
#        pass

cdef extern from "ctype.h":
    cdef int isspace( int )

cdef extern from "fileobject.h":
    cdef stdio.FILE *PyFile_AsFile(object) except NULL
    cdef int PyFile_CheckExact(object)


def scan(infile, num_to_read=None):
    """
    scan the file and return a numpy array of float64.

    :param infile: the file to scan
    :type infile: and open python file object.

    :param num_to_read=None: the number of values to read. If None,
                             then reads all the numbers in the file.
    :type num_to_read: integer

    Raises an TypeError if there are fewer than num_to_read numbers in the file.
    All text in the file that is not part of a floating point number is
    skipped over.

    After reading num_to_read numbers, the file is left before the next
    non-whitespace character in the file. This will often leave the file
    at the start of the next line, after scanning a line full of numbers.
    """

    cdef uint32_t  N, num_read, j

    N = UINT32_MAX if num_to_read is None else num_to_read

    ## does all this checking cost too much?
    ## and CheckExact is there later anyway...
    if  ( (type(infile) is not file) or
          infile.closed or
          not ('r' in infile.mode or 'a' in infile.mode)
        ):
        raise TypeError("infile must be an open file object")

    ## now to grab the C file handle
    cdef stdio.FILE* fp
    #cdef PyObject* py_file
    if PyFile_CheckExact(infile):
        fp = PyFile_AsFile(infile)
    else:
        raise TypeError("infile must be an open python file object")

    sys.stdout.flush()

    ## and do the actual work!
    cdef int c
    cdef double value
    cdef char* format_string = "%lg"

    cdef cnp.ndarray[double, ndim=1, mode="c"] out_arr
    if N == UINT32_MAX:
        # allocate an arbitarily small array
        # -- not too small, don't want to waste time making new arrays
        out_arr = np.zeros((128,), dtype= np.float64)
    else:
        out_arr = np.zeros((N,), dtype= np.float64)

    num_read = 1
    while num_read <= N:
        ## try to read a number
        ## keep advancing char by char until you get one
        while True:
            j = stdio.fscanf(fp, format_string, &value)
            if j == 0:
                c = stdio.fgetc(fp)
                continue
            break
        if j == stdio.EOF:
            break
        if num_read > out_arr.shape[0]: # need to make the array bigger
            # NOTE: ndarray.resize does not work in Cython
            # out_arr.resize( ( <int> out_arr.shape[0]*1.2, ), refcheck=False)
            temp = np.zeros( (num_read+<int> out_arr.shape[0]*1.5) )
            temp[:num_read-1] = out_arr
            out_arr = temp
        out_arr[num_read-1] = value
        num_read += 1

    num_read -= 1 # remove the extra tacked on at the end

    if N != UINT32_MAX and num_read < N:
        raise ValueError("not enough values in the file -- only read %i"%num_read)

    # advance past any whitespace left
    while True:
        c = stdio.fgetc(fp)
        if not isspace(c):
            # move back one char
            if c >-1: # not EOF
                stdio.fseek(fp, -1, stdio.SEEK_CUR)
            break

    # resize to fit:
    if out_arr.shape[0] > num_read:
        # resize can work if you don't need cython to access the data
        out_arr.resize( (num_read, ), refcheck=False )
    return out_arr


@cython.boundscheck(False)
def resize_test():
    """
    test of bounds_check code in face of re-size
    """
    cdef cnp.ndarray[double, ndim=1, mode="c"] arr

    arr = np.zeros( (1,) )
    arr[0] = 3.14
    arr.resize((4,), refcheck = False)
    arr[1] = 5.6
    arr[2] = 7.1
    arr[3] = 4.3
    return arr













