cdef extern from "Basics.h":
    pass

cdef extern from "TypeDefs.h":
    ctypedef unsigned long LETYPE    
    ctypedef unsigned long Seconds
    ctypedef unsigned char    Boolean
    ctypedef short    OSErr
    ctypedef unsigned long LETYPE

cdef extern from "CMYLIST.H":
    cdef cppclass CMyList:
        CMyList(long)
        OSErr AppendItem(char *)
        OSErr IList()

cdef extern from "LEList_c.h":
    cdef cppclass LEList_c:
        long numOfLEs
        LETYPE fLeType

cdef extern from "OLEList_c.h":
    cdef cppclass OLEList_c(LEList_c):
        pass
        
cdef extern from "GEOMETRY.H":
    ctypedef struct WorldPoint:
        float pLong
        float pLat
    ctypedef struct WorldPoint3D:
        WorldPoint p
        double z
    ctypedef struct WorldRect:
        long loLong
        long loLat
        long hiLong
        long hiLat

cdef extern from "TypeDefs.h":
    ctypedef struct TModelDialogVariables:
        Boolean bUncertain
        Boolean preventLandJumping
        
    ctypedef struct TR_OPTIMZE:
        Boolean isOptimizedForStep
        Boolean isFirstStep
        double value
        double uncertaintyValue

    ctypedef struct LERec:
        long leUnits
        long leKey
        long leCustomData
        WorldPoint p
        double z
        unsigned long releaseTime
        double ageInHrsWhenReleased
        unsigned long clockRef
        short pollutantType
        double mass
        double density
        double windage
        long dropletSize
        short dispersionStatus
        double riseVelocity
        short statusCode
        WorldPoint lastWaterPt
        unsigned long beachTime
        
    ctypedef struct VelocityRec:
        double u
        double v
    ctypedef struct LEWindUncertainRec:
        float downStream
        float crossStream
        
    ctypedef enum:
        OILSTAT_NOTRELEASED = 0
        OILSTAT_INWATER = 2
        OILSTAT_ONLAND
        OILSTAT_OFFMAPS = 7
        OILSTAT_EVAPORATED = 10 
    
    ctypedef enum:
        DONT_DISPERSE 
        DISPERSE
        HAVE_DISPERSED
        DISPERSE_NAT
        HAVE_DISPERSED_NAT
        EVAPORATE
        HAVE_EVAPORATED
        REMOVE
        HAVE_REMOVED

cdef extern from "Random_c.h":
    cdef cppclass Random_c:
        Boolean bUseDepthDependent
        double fDiffusionCoefficient
        double fUncertaintyFactor
        TR_OPTIMZE fOptimize            
        WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)
        OSErr        PrepareForModelStep()
        void        ModelStepIsDone()
cdef extern from "WindMover_c.h":
    cdef cppclass WindMover_c:
        double fSpeedScale
        double fAngleScale
        double fMaxSpeed
        double fMaxAngle
        double fSigma2
        double fSigmaTheta
        unsigned long fUncertainStartTime
        unsigned long fDuration
        Boolean bUncertaintyPointOpen
        Boolean bSubsurfaceActive
        double fGamma
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        LEWindUncertainRec **fWindUncertaintyList
        long **fLESetSizes
        WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)
        OSErr        PrepareForModelStep()
        OSErr        AllocateUncertainty()

cdef extern from "GridVel_c.h":
    cdef cppclass GridVel_c:
        pass

cdef extern from "Map_c.h":
    cdef cppclass Map_c:
        pass
        
cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        pass
        
cdef extern from "Mover_c.h":
    cdef cppclass Mover_c:
        pass
        
cdef extern from "CurrentMover_c.h":
    cdef cppclass CurrentMover_c(Mover_c):
        pass

cdef extern from "ShioTimeValue_c.h":
    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c()
        OSErr    ReadTimeValues (char *path)
        WorldPoint GetRefWorldPoint()
        
cdef extern from "Model_c.h":
    cdef cppclass Model_c:
        Model_c()
        void SetStartTime(Seconds)
        void SetDuration(Seconds)
        void SetModelTime(Seconds)
        void SetTimeStep(Seconds)
        Seconds GetStartTime()
        Seconds GetModelTime()
        Seconds GetTimeStep()
        CMyList *LESetsList
        TModelDialogVariables fDialogVariables

cdef extern from "CATSMover_c.h":
    ctypedef struct TCM_OPTIMZE:
        Boolean isOptimizedForStep
        Boolean isFirstStep
        double     value
        
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
        OSErr           ComputeVelocityScale()
        void        SetTimeDep(OSSMTimeValue_c *time_dep)
        OSErr        PrepareForModelStep()
        void        ModelStepIsDone()

cdef public enum type_defs:
    status_not_released = OILSTAT_NOTRELEASED
    status_in_water = OILSTAT_INWATER
    status_on_land = OILSTAT_ONLAND
    status_off_maps = OILSTAT_OFFMAPS
    status_evaporated = OILSTAT_EVAPORATED
    disp_status_dont_disperse = DONT_DISPERSE
    disp_status_disperse = DISPERSE
    disp_status_have_dispersed = HAVE_DISPERSED
    disp_status_disperse_nat = DISPERSE_NAT
    disp_status_have_dispersed_nat = HAVE_DISPERSED_NAT
    disp_status_evaporate = EVAPORATE
    disp_status_have_evaporated = HAVE_EVAPORATED
    disp_status_remove = REMOVE
    disp_status_have_removed = HAVE_REMOVED
