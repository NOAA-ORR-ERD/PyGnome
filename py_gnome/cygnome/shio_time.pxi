include "ossm_time.pxi"

cdef extern from "ShioTimeValue_c.h":
    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c()
        OSErr    ReadTimeValues (char *path)
        WorldPoint GetRefWorldPoint()
