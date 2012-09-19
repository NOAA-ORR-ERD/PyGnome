cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        pass

cdef extern from "ShioTimeValue_c.h":
    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c()
        OSErr    ReadTimeValues (char *path)
        WorldPoint GetRefWorldPoint()
