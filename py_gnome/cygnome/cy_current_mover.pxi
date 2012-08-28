include "mover.pxi"
cdef extern from "CurrentMover_c.h":
    cdef cppclass CurrentMover_c(Mover_c):
        pass
