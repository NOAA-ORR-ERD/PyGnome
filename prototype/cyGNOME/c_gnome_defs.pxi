
ctypedef unsigned char    Boolean

cdef extern from "Earl.h":
    pass

cdef extern from "TypeDefs.h":
    ctypedef unsigned long LETYPE    
    ctypedef unsigned long Seconds

cdef extern from "GEOMETRY.H":
    ctypedef struct WorldPoint:
        long pLong
        long pLat
    ctypedef struct WorldPoint3D:
        WorldPoint p
        double z
    ctypedef struct WorldRect:
        long loLong
        long loLat
        long hiLong
        long hiLat

cdef extern from "TypeDefs.h":
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

cdef extern from "CROSS.H":
    pass
cdef extern from "OSSM.H":
    pass
cdef extern from "CarbonUtil.h":
    cdef int gSessionDocumentIsOpen

cdef extern from "CLASSES.H":
    ctypedef struct TR_OPTIMZE:
        Boolean isOptimizedForStep
        Boolean isFirstStep
        double value
        double uncertaintyValue

cdef extern from "Random/Random_c.h":
    cdef cppclass Random_c:
        Boolean bUseDepthDependent
        double fDiffusionCoefficient
        double fUncertaintyFactor
        TR_OPTIMZE fOptimize            
        WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)

cdef extern from "WindMover/WindMover_c.h":
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
        
