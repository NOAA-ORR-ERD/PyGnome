IF not HEADERS.count("_OSSM_TIME_"):
    DEF HEADERS = HEADERS + ["_OSSM_TIME_"]
    cdef extern from "OSSMTimeValue_c.h":
        cdef cppclass OSSMTimeValue_c:
            pass
