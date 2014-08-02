"""
Class serves as a base class for cython wrappers around C++ xxx.
"""
from grids cimport TimeGridVel_c, TimeGridWindRect_c, TimeGridWindCurv_c
#from type_defs cimport OSErr, LoadedData, VelocityFH


cdef extern from *:
    TimeGridWindRect_c* dc_base_to_rect "dynamic_cast<TimeGridWindRect_c *>" \
        (TimeGridVel_c *) except NULL
    TimeGridWindCurv_c* dc_base_to_curv "dynamic_cast<TimeGridWindCurv_c *>" \
        (TimeGridVel_c *) except NULL

cdef class CyTimeGridVel:
    cdef TimeGridVel_c * timegrid
