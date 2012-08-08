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
        LEWindUncertainRec **fWindUncertaintyList
        long **fLESetSizes
        double breaking_wave_height
        double mix_layer_depth
        OSErr get_move(int, unsigned long, unsigned long, char *, char *, char *, char *, double, double, double, double, char *, char*, int)
        OSErr get_move(int, unsigned long, unsigned long, char *, char *, char *, char *, double, double, char*, int)
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool)