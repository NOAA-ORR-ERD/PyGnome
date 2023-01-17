import cython

from .grids cimport TimeGridWindRect_c
from .cy_grid cimport CyTimeGridVel, dc_base_to_rect


@cython.final
cdef class CyTimeGridWindRect(CyTimeGridVel):
    '''
    cython wrapper around TimeGridWindRect_c C++ class
    '''
    def __cinit__(self):
        if type(self) == CyTimeGridWindRect:
            self.timegrid = new TimeGridWindRect_c()
            self.timegridwind = dc_base_to_rect(self.timegrid)

    def __init__(self, datafile, topology=None):
        # can we move this to base class?
        '''
        :param datafile: PathLike object to load the data from

        :param topology=None: PathLike object to load topology data.
        '''
        self.load_data(datafile, topology)

    def __dealloc__(self):
        # since this is allocated in this class, free memory here as well
        if self.timegrid is not NULL:
            del self.timegrid
            self.timegrid = NULL
            self.timegridwind = NULL
