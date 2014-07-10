"""
C++ GridVel and TimeGridVel classes. Write Cython wrappers around these objects
so Wind and Current objects can be initialized in Python independent of movers
"""

from type_defs cimport (OSErr,
                        LoadedData,
                        VelocityFH,
                        VelocityRec,
                        WorldPoint3D,
                        Seconds)


'''
Keep following two definitions for now so current cython movers code works
'''
cdef extern from "GridVel_c.h":
    cdef cppclass GridVel_c:
        pass

cdef extern from "TimeGridVel_c.h":
    cdef cppclass TimeGridVel_c:
        #======================================================================
        # OSErr                TextRead(char *path, char *topFilePath)
        # OSErr                ReadInputFileNames(char *fileNamesPath)
        # OSErr                ReadTimeData(long index, VelocityFH *velocityH, char* errmsg)
        # void                 DisposeLoadedData(LoadedData * dataPtr)
        # void                 ClearLoadedData(LoadedData * dataPtr)
        # void                 DisposeAllLoadedData()
        #======================================================================
        OSErr       TextRead(char *path, char *topFilePath)
        OSErr       ReadInputFileNames(char *fileNamesPath)
        OSErr       SetInterval(char *errmsg, const Seconds& model_time)
        VelocityRec GetScaledPatValue(Seconds& time, WorldPoint3D p)

    cdef cppclass TimeGridWindRect_c(TimeGridVel_c):
        pass