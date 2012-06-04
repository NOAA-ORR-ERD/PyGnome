IF not HEADERS.count("_GRID_VEL_"):
    DEF HEADERS = HEADERS + ["_GRID_VEL_"]
    cdef extern from "GridVel_c.h":
        cdef cppclass GridVel_c:
            pass
