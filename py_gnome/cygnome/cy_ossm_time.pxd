"""
cy_ossm_time.pyx module declaration file
Used to share members of the CyOSSMTime class
"""

cdef class CyOSSMTime:
    cdef OSSMTimeValue_c * time_dep