IF not HEADERS.count("_CURRENT_MOVER_"):
    DEF HEADERS = HEADERS + ["_CURRENT_MOVER_"]
    include "mover.pxi"
    cdef extern from "CurrentMover_c.h":
        cdef cppclass CurrentMover_c(Mover_c):
            pass
