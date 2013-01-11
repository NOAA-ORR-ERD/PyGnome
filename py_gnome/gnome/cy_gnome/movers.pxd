"""
Declare the C++ mover classes from lib_gnome
"""

from type_defs cimport *
from utils cimport OSSMTimeValue_c

from libcpp cimport bool

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
        OSErr                TextRead(char *path,char *topFilePath)
        OSErr                ReadTimeData(long index,VelocityFH *velocityH, char* errmsg)
        void                 DisposeLoadedData(LoadedData * dataPtr)
        void                 ClearLoadedData(LoadedData * dataPtr)
        void                 DisposeAllLoadedData()
    cdef cppclass TimeGridVelRect_c:
        OSErr                TextRead(char *path,char *topFilePath)
    cdef cppclass TimeGridVelCurv_c:
        OSErr                TextRead(char *path,char *topFilePath)
    cdef cppclass TimeGridWindRect_c:
        OSErr                TextRead(char *path,char *topFilePath)
    cdef cppclass TimeGridWindCurv_c:
        OSErr                TextRead(char *path,char *topFilePath)

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
        pass

cdef extern from "CurrentMover_c.h":
    cdef cppclass CurrentMover_c(Mover_c):
        pass

cdef extern from "WindMover_c.h":
    cdef cppclass WindMover_c:
        WindMover_c() except +
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        double fDuration
        double fUncertainStartTime
        double fSpeedScale
        double fAngleScale
        
        OSErr PrepareForModelRun()
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID)
        void SetTimeDep(OSSMTimeValue_c *ossm)
        OSErr GetTimeValue(Seconds &time, VelocityRec *vel)
        OSErr PrepareForModelStep(Seconds &time, Seconds &time_step, bool uncertain, int numLESets, int* LESetsSizesList)	# currently this happens in C++ get_move command
        void ModelStepIsDone()
        
        
        
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
        #=======================================================================
        # WorldPoint      refP                
        # GridVel_c       *fGrid    
        # long            refZ                     
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
       
        #int   ReadTopology(char* path, Map_c **newMap)    # what is this for? Do we want to expose? What is Map_c?
        void  SetRefPosition (WorldPoint , long )    # Could we use WorldPoint3D for this?
        #OSErr ComputeVelocityScale(Seconds&)    # seems to require TMap, TCATSMover
       
        OSErr PrepareForModelRun()
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        void  SetTimeDep(OSSMTimeValue_c *ossm)
        OSErr PrepareForModelStep(Seconds &time, Seconds &time_step, bool uncertain, int numLESets, int* LESetsSizesList)    # currently this happens in C++ get_move command
        void  ModelStepIsDone()
       

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
        WorldPoint3D        GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        OSErr 		PrepareForModelRun()
        OSErr 		get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        OSErr 		PrepareForModelStep(Seconds&, Seconds&, bool, int numLESets, int* LESetsSizesList)
        void 		ModelStepIsDone()
        void 		SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr                TextRead(char *path,char *topFilePath)
        
cdef extern from "GridWindMover_c.h":
    
    cdef cppclass GridWindMover_c:
        TimeGridVel_c    *timeGrid
        Boolean fIsOptimizedForStep
        float    fWindScale
        float    fArrowScale
        short    fUserUnits
        
        GridWindMover_c ()
        WorldPoint3D        GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        OSErr 		PrepareForModelRun()
        OSErr 		get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spillID)
        OSErr 		PrepareForModelStep(Seconds&, Seconds&, bool, int numLESets, int* LESetsSizesList)
        void 		ModelStepIsDone()
        void 		SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr                TextRead(char *path,char *topFilePath)
        
cdef extern from "Random_c.h":
    cdef cppclass Random_c:
        Boolean bUseDepthDependent
        double fDiffusionCoefficient
        double fUncertaintyFactor
        TR_OPTIMZE fOptimize
        OSErr PrepareForModelRun()
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool, int numLESets, int* LESetsSizesList)
        void ModelStepIsDone()
