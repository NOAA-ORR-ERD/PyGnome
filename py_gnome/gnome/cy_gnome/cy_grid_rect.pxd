"""
Class serves as a base class for cython wrappers around C++ xxx.
"""
from grids cimport TimeGridVel_c
from cy_grid cimport CyTimeGridVel
from grids cimport TimeGridWindRect_c


cdef class CyTimeGridWindRect(CyTimeGridVel):
    cdef TimeGridWindRect_c * timegridwind
