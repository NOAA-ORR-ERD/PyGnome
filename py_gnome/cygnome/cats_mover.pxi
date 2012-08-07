include "type_defs.pxi"
include "current_mover.pxi"
include "grid_vel.pxi"

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
        OSErr           ComputeVelocityScale(Seconds&, Seconds&, Seconds&)
        void        SetTimeDep(OSSMTimeValue_c *time_dep)
        OSErr        PrepareForModelStep(Seconds&, Seconds&, Seconds&, Seconds&, bool)
        void        ModelStepIsDone()
        OSErr		get_move(int, unsigned long, unsigned long, unsigned long, unsigned long, char *, char *, char *)
        OSErr		get_move(int, unsigned long, unsigned long, unsigned long, unsigned long, char *, char *)
