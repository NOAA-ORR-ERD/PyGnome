include "type_defs.pxi"

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
        OSSMTimeValue_c *timeDep
        LEWindUncertainRec **fWindUncertaintyList
        long **fLESetSizes
        OSErr get_move(int, unsigned long, unsigned long, char *, char *, char *, double, double, char *)
        OSErr get_move(int, unsigned long, unsigned long, char *, char *, char *)
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool)
        void SetTimeDep(OSSMTimeValue_c *time_dep)