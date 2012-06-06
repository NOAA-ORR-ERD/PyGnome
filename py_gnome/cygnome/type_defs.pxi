IF not HEADERS.count("_TYPE_DEFS_"):
    DEF HEADERS = HEADERS + ["_TYPE_DEFS_"] 

    cdef extern from "Basics.h":
        pass
        
    cdef extern from "TypeDefs.h":
        ctypedef unsigned long LETYPE    
        ctypedef unsigned long Seconds
        ctypedef unsigned char    Boolean
        ctypedef short    OSErr
        ctypedef unsigned long LETYPE
    
    cdef extern from "GEOMETRY.H":
        ctypedef struct WorldPoint:
            double pLong
            double pLat
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


