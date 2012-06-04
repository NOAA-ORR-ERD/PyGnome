IF not HEADERS.count("_MODEL_"):
    DEF HEADERS = HEADERS + ["_MODEL_"]
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
