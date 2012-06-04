IF not HEADERS.count("_RANDOM_MOVER_"):
    DEF HEADERS = HEADERS + ["_RANDOM_MOVER_"]
    cdef extern from "Random_c.h":
        cdef cppclass Random_c:
            Boolean bUseDepthDependent
            double fDiffusionCoefficient
            double fUncertaintyFactor
            TR_OPTIMZE fOptimize            
            WorldPoint3D GetMove (Seconds timeStep, long setIndex, long leIndex, LERec *theLE, LETYPE leType)
            OSErr        PrepareForModelStep()
            void        ModelStepIsDone()
