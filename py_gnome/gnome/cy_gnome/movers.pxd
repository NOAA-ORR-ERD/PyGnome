"""
Declare the C++ mover classes from lib_gnome
"""
from libcpp cimport bool

from libc.stdint cimport int32_t

from type_defs cimport *
from utils cimport OSSMTimeValue_c  

"""
Following are used by some of the methods of the movers
Currently, it looks like all methods are exposed from C++ so
these are declared here. This may get smaller as we work through the
cython files since there maybe no need to expose all the C++ functionality.
"""    
cdef extern from "GridVel_c.h":
    cdef cppclass GridVel_c:
        pass

cdef extern from "TimeGridVel_c.h":
    cdef cppclass TimeGridVel_c:
        OSErr                TextRead(char *path, char *topFilePath)
        OSErr                ReadInputFileNames(char *fileNamesPath)    # not used by pyx -- just used for some testing right now
        OSErr                ReadTimeData(long index, VelocityFH *velocityH, char* errmsg)
        void                 DisposeLoadedData(LoadedData * dataPtr)
        void                 ClearLoadedData(LoadedData * dataPtr)
        void                 DisposeAllLoadedData()
    
# TODO: pre-processor directive for cython, but what is its purpose?
# comment for now so it doesn't give compile time errors - not sure LELIST_c is used anywhere either
#IF not HEADERS.count("_LIST_"):
#    DEF HEADERS = HEADERS +  ["_LE_LIST_"]
cdef extern from "LEList_c.h":
    cdef cppclass LEList_c:
        long numOfLEs
        LETYPE fLeType

cdef extern from "Map_c.h":
    cdef cppclass Map_c:
        pass

"""
movers:
"""
cdef extern from "Mover_c.h":
    cdef cppclass Mover_c:
        OSErr PrepareForModelRun()
        OSErr PrepareForModelStep(Seconds &time, Seconds &time_step,
                                  bool uncertain, int numLESets, int32_t *LESetsSizesList)    # currently this happens in C++ get_move command
        void ModelStepIsDone()
        OSErr ReallocateUncertainty(int numLEs, short* LE_status)

cdef extern from "CurrentMover_c.h":
    cdef cppclass CurrentMover_c(Mover_c):
        pass

cdef extern from "WindMover_c.h":
    cdef cppclass WindMover_c(Mover_c):
        WindMover_c() except +
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        double fDuration
        double fUncertainStartTime
        double fSpeedScale
        double fAngleScale
        
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID)
        void SetTimeDep(OSSMTimeValue_c *ossm)
        OSErr GetTimeValue(Seconds &time, VelocityRec *vel)
        
cdef extern from "Random_c.h":
    cdef cppclass Random_c(Mover_c):
        Random_c() except +
        double fDiffusionCoefficient
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)        
        
cdef extern from "RandomVertical_c.h":
    cdef cppclass RandomVertical_c(Mover_c):
        RandomVertical_c() except +
        double fVerticalDiffusionCoefficient
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)        
        
cdef extern from "RiseVelocity_c.h":
    cdef cppclass RiseVelocity_c(Mover_c):
        RiseVelocity_c() except +
        double water_density
        double water_viscosity
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len,
                       WorldPoint3D* ref, WorldPoint3D* delta,
                       double* rise_velocity, double* density, double* droplet_size,
                       short* LE_status, LEType spillType, long spillID)        
        
cdef extern from "CATSMover_c.h":
   #============================================================================
   # ctypedef struct TCM_OPTIMZE:
   #    Boolean isOptimizedForStep
   #    Boolean isFirstStep
   #    double  value
   #============================================================================
   cdef cppclass CATSMover_c(CurrentMover_c):
        CATSMover_c() except +
        double          fEddyDiffusion
        short           scaleType                 
        double          scaleValue
        Boolean         bTimeFileActive
        WorldPoint      refP
        long            refZ
        #=======================================================================
        # GridVel_c       *fGrid                         
        # char            scaleOtherFile[32]
        # double          refScale
        # Boolean         bRefPointOpen
        # Boolean         bUncertaintyPointOpen
        # Boolean         bTimeFileOpen
        # Boolean         bShowGrid
        # Boolean         bShowArrows
        # double          arrowScale
        # OSSMTimeValue_c *timeDep
        # double          fEddyV0
        #=======================================================================
        #TCM_OPTIMZE     fOptimize
       
        int   TextRead(char* path)
        void  SetRefPosition (WorldPoint p, long z)    # Could we use WorldPoint3D for this?
        #OSErr ComputeVelocityScale(Seconds&)    # seems to require TMap, TCATSMover
       
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        void  SetTimeDep(OSSMTimeValue_c *ossm)
       

cdef extern from "GridCurrentMover_c.h":
    
    cdef struct GridCurrentVariables:
        char        *pathName
        char        *userName
        double     alongCurUncertainty
        double     crossCurUncertainty
        double     uncertMinimumInMPS
        double     curScale
        double     startTimeInHrs
        double     durationInHrs
        short        gridType
        Boolean     bShowGrid
        Boolean     bShowArrows
        Boolean    bUncertaintyPointOpen
        double     arrowScale
        double     arrowDepth

    cdef cppclass GridCurrentMover_c(CurrentMover_c):
        GridCurrentVariables fVar
        TimeGridVel_c    *timeGrid
        Boolean fIsOptimizedForStep
        Boolean fAllowVerticalExtrapolationOfCurrents
        float    fMaxDepthForExtrapolation
        
        GridCurrentMover_c ()
        WorldPoint3D    GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        OSErr 		    get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        void 		    SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr           TextRead(char *path,char *topFilePath)
        OSErr          ExportTopology(char *topFilePath)
        
cdef extern from "GridWindMover_c.h":
    
    cdef cppclass GridWindMover_c(WindMover_c):
        TimeGridVel_c    *timeGrid
        Boolean fIsOptimizedForStep
        float    fWindScale
        float    fArrowScale
        short    fUserUnits
        
        GridWindMover_c ()
        WorldPoint3D    GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        void 		    SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr           TextRead(char *path,char *topFilePath)
        OSErr          ExportTopology(char *topFilePath)
        
cdef extern from "GridMap_c.h":
    
    cdef cppclass GridMap_c:
        GridVel_c    *fGrid
        WorldRect fMapBounds
        LONGH    fBoundarySegmentsH
        LONGH    fBoundaryTypeH
        LONGH    fBoundaryPointsH
        
        GridMap_c ()
        LONGH    GetBoundarySegs()
        LONGH    GetWaterBoundaries()
        LONGH    GetBoundaryPoints()
        OSErr 		    ExportTopology(char *path)
        OSErr 		    SaveAsNetCDF(char *path)
        OSErr           TextRead(char *path)
        
