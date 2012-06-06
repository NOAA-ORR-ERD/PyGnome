IF not HEADERS.count("_WIND_MOVER_"):
    DEF HEADERS = HEADERS + ["_WIND_MOVER_"]
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
