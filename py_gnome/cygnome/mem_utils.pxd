"""
MemUtils functions available from lib_gnome 
"""

from type_defs cimport *

cdef extern from "MemUtils.h":
    Handle _NewHandle(long )
    void _DisposeHandleReally(Handle)
    long _GetHandleSize(Handle)
