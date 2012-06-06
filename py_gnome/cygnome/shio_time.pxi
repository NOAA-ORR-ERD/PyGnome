IF not HEADERS.count("_SHIO_TIME_"):
    DEF HEADERS = HEADERS + ["_SHIO_TIME_"]
    cdef extern from "ShioTimeValue_c.h":
        cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
            ShioTimeValue_c()
            OSErr    ReadTimeValues (char *path)
            WorldPoint GetRefWorldPoint()
