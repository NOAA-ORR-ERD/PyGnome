"""
cy_ossm_time.pyx module declaration file
Used to share members of the CyOSSMTime class
This file must have the same name as the pyx file, but with a pxd suffix
"""

from utils cimport OSSMTimeValue_c, ShioTimeValue_c


# Define user units for velocity. In C++, these are #defined as follows.
# move these here so we can access them from python without instantiating
# a CyOSSMTime object
_user_units_dict = {-1: 'undefined',
                    1: 'knots',
                    2: 'meters per second',
                    3: 'miles per hour'}


cdef class CyOSSMTime:
    cdef OSSMTimeValue_c * time_dep
    cdef object _user_units_dict
    cdef public int _file_format


# dynamic cast ossm object to shio object
# cy_shio_time extends cy_ossm_time
cdef extern from *:
    ShioTimeValue_c* dynamic_cast_ptr "dynamic_cast<ShioTimeValue_c *>" \
        (OSSMTimeValue_c *) except NULL