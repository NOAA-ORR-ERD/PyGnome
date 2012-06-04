IF not HEADERS.count("_MOVER_"):
    DEF HEADERS = HEADERS + ["_MOVER_"]
    cdef extern from "Mover_c.h":
        cdef cppclass Mover_c:
            pass
