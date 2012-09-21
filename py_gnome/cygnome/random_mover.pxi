cdef extern from "Random_c.h":
    cdef cppclass Random_c:
        Boolean bUseDepthDependent
        double fDiffusionCoefficient
        double fUncertaintyFactor
        TR_OPTIMZE fOptimize
        OSErr get_move(int, unsigned long, unsigned long, char *, char *, double)
        OSErr get_move(int, unsigned long, unsigned long, char *, char *)
        WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)
        OSErr PrepareForModelStep(Seconds&, Seconds&, bool)
        void ModelStepIsDone()
