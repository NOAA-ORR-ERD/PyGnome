import cython

from grids cimport TimeGridWindCurv_c
from cy_grid cimport dc_base_to_curv
from cy_grid_rect cimport CyTimeGridWindRect


@cython.final
cdef class CyTimeGridWindCurv(CyTimeGridWindRect):
    cdef TimeGridWindCurv_c * timegridwindcurv

    def __cinit__(self):
        if type(self) == CyTimeGridWindCurv:
            '''
            Added the if check above incase this class gets derived in the
            future
            '''
            self.timegrid = new TimeGridWindCurv_c()
            self.timegridwindcurv = dc_base_to_curv(self.timegrid)

    def __dealloc__(self):
        # since this is allocated in this class, free memory here as well
        if self.timegrid is not NULL:
            del self.timegrid
            self.timegrid = NULL
            self.timegridwindcurv = NULL
