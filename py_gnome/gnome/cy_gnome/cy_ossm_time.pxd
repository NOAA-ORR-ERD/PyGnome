"""
cy_ossm_time.pyx module declaration file
Used to share members of the CyOSSMTime class
This file must have the same name as the pyx file, but with a pxd suffix
"""

from ossm_time cimport OSSMTimeValue_c

cdef class CyOSSMTime:
    cdef OSSMTimeValue_c * time_dep