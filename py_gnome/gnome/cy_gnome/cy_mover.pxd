import cython
from movers cimport Mover_c, WindMover_c
'''
Define base classes that keep common code here. These classes don't create
C++ objects; the derived classes create the C++ objects. These base classes
just expect the objects to exist

Class serves as a base class for cython wrappers around C++ movers.
The C++ movers derive from Mover_c.cpp. CyMover defines the mover pointer, but
each class that derives from CyMover must instantiate this object to be either
a WindMover_c, RandomMover_c, and so forth
'''
cdef class CyMover:
    cdef Mover_c * mover


cdef class CyWindMoverBase(CyMover):
    cdef WindMover_c * wind
