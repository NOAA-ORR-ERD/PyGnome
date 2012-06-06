IF not HEADERS.count("_MAP_"):
    DEF HEADERS = HEADERS + ["_MAP_"]
    cdef extern from "Map_c.h":
        cdef cppclass Map_c:
            pass
