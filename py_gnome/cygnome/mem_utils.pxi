"""

"""

cdef extern from "MemUtils.h":
    Handle _NewHandle(long )
    void _DisposeHandleReally(Handle)
    long _GetHandleSize(Handle)
