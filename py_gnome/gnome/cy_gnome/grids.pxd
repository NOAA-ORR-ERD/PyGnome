"""
C++ GridVel and TimeGridVel classes. Write Cython wrappers around these objects
so Wind and Current objects can be initialized in Python independent of movers
"""

from libcpp cimport bool

from type_defs cimport (OSErr,
                        LoadedData,
                        VelocityFH,
                        VelocityRec,
                        WorldPoint3D,
                        Seconds,
                        WorldRect,
                        LongPointHdl,
                        TopologyHdl,
                        DAGHdl,
                        LONGH)


'''
Keep following two definitions for now so current cython movers code works
'''
cdef extern from "GridVel_c.h":
    cdef cppclass GridVel_c:
        pass

cdef extern from "DagTree.h":
    cdef cppclass TDagTree:
        LongPointHdl	GetPointsHdl()
        TopologyHdl		GetTopologyHdl()
        VelocityFH		GetVelocityHdl()
        DAGHdl			GetDagTreeHdl()

cdef extern from "TriGridVel_c.h":
    cdef cppclass TriGridVel_c:
        TDagTree *fDagTree
        LongPointHdl GetPointsHdl()
        TopologyHdl GetTopologyHdl()
        VelocityFH GetVelocityHdl()

cdef extern from "TimeGridVel_c.h":
    cdef cppclass TimeGridVel_c:
        long       	fTimeShift
        bool    	fAllowExtrapolationInTime
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
        OSErr 		get_values(int n, Seconds model_time, WorldPoint3D* ref, VelocityRec* vels)

    cdef cppclass TimeGridWindRect_c(TimeGridVel_c):
        pass

    cdef cppclass TimeGridWindCurv_c(TimeGridWindRect_c):
        pass


cdef extern from "GridMap_c.h":
    cdef cppclass GridMap_c:
        GridVel_c   *fGrid
        WorldRect   fMapBounds
        LONGH       fBoundarySegmentsH
        LONGH       fBoundaryTypeH
        LONGH       fBoundaryPointsH

        GridMap_c ()
        LONGH       GetBoundarySegs()
        LONGH       GetWaterBoundaries()
        LONGH       GetBoundaryPoints()
        OSErr       ExportTopology(char *path)
        OSErr       SaveAsNetCDF(char *path)
        OSErr       TextRead(char *path)
