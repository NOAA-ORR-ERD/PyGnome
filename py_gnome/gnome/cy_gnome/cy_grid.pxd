"""
Class serves as a base class for cython wrappers around C++ xxx.
"""
from grids cimport TimeGridVel_c, TimeGridWindRect_c
from type_defs cimport OSErr, LoadedData, VelocityFH

cdef class CyTimeGridVel:
    cdef TimeGridVel_c * timegrid
