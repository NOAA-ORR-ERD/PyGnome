IF not HEADERS.count("_CATS_MOVER_"):
    DEF HEADERS = HEADERS + ["_CATS_MOVER_"]
    include "type_defs.pxi"
    include "current_mover.pxi"
    include "grid_vel.pxi"
    include "ossm_time.pxi"
    
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

