"""
cy_shio_time.pyx module declaration file
Used to share members of the CyShioTime class
This file must have the same name as the pyx file, but with a pxd suffix
"""

from utils cimport ShioTimeValue_c

cdef class CyShioTime:
    cdef ShioTimeValue_c * shio
    