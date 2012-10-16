"""
Declare the C++ mover classes from lib_gnome
"""

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


# TODO: pre-processor directive for cython, but what is its purpose?
# comment for now so it doesn't give compile time errors
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
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        LEWindUncertainRec **fWindUncertaintyList
        long **fLESetSizes
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID)
        void SetTimeDep(OSSMTimeValue_c *)
        # ARE FOLLOWING USED IN CYTHON??
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool)	# currently this happens in C++ get_move command
        
cdef extern from "CATSMover_c.h":
    ctypedef struct TCM_OPTIMZE:
        Boolean isOptimizedForStep
        Boolean isFirstStep
        double  value
        
    cdef cppclass CATSMover_c(CurrentMover_c):
        WorldPoint         refP                
        GridVel_c        *fGrid    
        long             refZ                     
        short             scaleType                 
        double             scaleValue             
        char             scaleOtherFile[32]
        double             refScale
        Boolean         bRefPointOpen
        Boolean            bUncertaintyPointOpen
        Boolean         bTimeFileOpen
        Boolean            bTimeFileActive
        Boolean         bShowGrid
        Boolean         bShowArrows
        double             arrowScale
        OSSMTimeValue_c *timeDep
        double            fEddyDiffusion    
        double            fEddyV0
        TCM_OPTIMZE     fOptimize
        WorldPoint3D    GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
        int             ReadTopology(char* path, Map_c **newMap)
        void            SetRefPosition (WorldPoint p, long z)
        OSErr           ComputeVelocityScale(Seconds&)
        void        SetTimeDep(OSSMTimeValue_c *time_dep)
        void        ModelStepIsDone()
        OSErr 		get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)

cdef extern from "NetCDFMover_c.h":
    
    cdef struct NetCDFVariables:
        char        *pathName
        char        *userName
        double     alongCurUncertainty
        double     crossCurUncertainty
        double     uncertMinimumInMPS
        double     curScale
        double     startTimeInHrs
        double     durationInHrs
        long        maxNumDepths
        short        gridType
        Boolean     bShowGrid
        Boolean     bShowArrows
        Boolean    bUncertaintyPointOpen
        double     arrowScale
        double     arrowDepth

    cdef cppclass NetCDFMover_c(CurrentMover_c):
        long fNumRows
        long fNumCols
        long fNumDepthLevels
        NetCDFVariables fVar
        Boolean bShowDepthContours
        Boolean bShowDepthContourLabels
        GridVel_c    *fGrid
        Seconds **fTimeHdl
        float **fDepthLevelsHdl
        float **fDepthLevelsHdl2
        float hc
        float **fDepthsH
        float fFillValue
        double fFileScaleFactor
        Boolean fIsNavy
        Boolean fIsOptimizedForStep
        Boolean fOverLap
        Seconds fOverLapStartTime
        long fTimeShift
        Boolean fAllowExtrapolationOfCurrentsInTime
        Boolean fAllowVerticalExtrapolationOfCurrents
        float    fMaxDepthForExtrapolation
        
        NetCDFMover_c ()
        WorldPoint3D        GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        OSErr                 ReadTimeData(long index,VelocityFH *velocityH, char* errmsg)
        void                 DisposeLoadedData(LoadedData * dataPtr)
        void                 ClearLoadedData(LoadedData * dataPtr)
        void                 DisposeAllLoadedData()
        OSErr        get_move(int, unsigned long, unsigned long, char *, char *, char *)
        OSErr        get_move(int, unsigned long, unsigned long, char *, char *)
        
cdef extern from "Random_c.h":
    cdef cppclass Random_c:
        Boolean bUseDepthDependent
        double fDiffusionCoefficient
        double fUncertaintyFactor
        TR_OPTIMZE fOptimize
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool)
        void ModelStepIsDone()
