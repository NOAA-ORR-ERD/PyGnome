import cython
DEF HEADERS = list()
include "type_defs.pxi"
include "c_mylist.pxi"
include "le_list.pxi"
include "model.pxi"

cdef extern Model_c *model

cpdef set_model_start_time(Seconds uh):
    model.SetStartTime(uh)
    
cpdef set_model_duration(Seconds uh):
    model.SetDuration(uh)
    
cpdef set_model_time(Seconds uh):
    model.SetModelTime(uh)
    
cpdef set_model_timestep(Seconds uh):
    model.SetTimeStep(uh)

cpdef step_model():
    cdef Seconds t, s
    t = model.GetModelTime()
    s = model.GetTimeStep()
    model.SetModelTime(t + s)

cpdef set_model_uncertain():
    model.fDialogVariables.bUncertain = True

cpdef initialize_model(spills):
    cdef CMyList *le_superlist
    cdef LEList_c *le_list
    cdef OSErr err
    le_superlist = new CMyList(sizeof(LEList_c*))
    le_superlist.IList()
    for spill in spills:
        le_type = spill.uncertain + 1
        le_list = new LEList_c()
        le_list.fLeType = le_type
        le_list.numOfLEs = len(spill.npra)
        err = le_superlist.AppendItem(<char *>&le_list) #hrm.
    model.LESetsList = le_superlist
