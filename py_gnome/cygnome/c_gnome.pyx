import cython
import random
import math

from libcpp.vector cimport vector
from cython.operator import preincrement as preinc

cimport numpy as np
import numpy as np

include "c_gnome_defs.pxi"

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
    cdef OLEList_c *le_list
    cdef OSErr err
    le_superlist = new CMyList(sizeof(OLEList_c*))
    le_superlist.IList()
    for spill in spills:
        le_type = spill.uncertain + 1
        le_list = new OLEList_c()
        le_list.fLeType = le_type
        le_list.numOfLEs = len(spill.npra)
        err = le_superlist.AppendItem(<char *>&le_list) #hrm.
    model.LESetsList = le_superlist

#====================================================================#
# cdef class shio_time_value:
#     
#     cdef ShioTimeValue_c *time_value
#     
#     def __cinit__(self):
#         self.time_value = new ShioTimeValue_c()
#         
#     def __dealloc__(self):
#         del self.time_value
#         
#     def __init__(self):
#        pass
#     
#     def read_time_values(self, path, format, units):
#         self.time_value.ReadTimeValues(path, format, units)
#====================================================================#

cdef class cats_mover:

    cdef CATSMover_c *mover
    
    def __cinit__(self):
        self.mover = new CATSMover_c()
    
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, scale_type, scale_value=1, diffusion_coefficient=1):
        cdef WorldPoint p
        self.mover.scaleType = scale_type
        self.mover.scaleValue = scale_value
        self.mover.fEddyDiffusion = diffusion_coefficient
        ## should not have to do this manually.
        ## make-shifting for now.
        self.mover.fOptimize.isOptimizedForStep = 0
        self.mover.fOptimize.isFirstStep = 1  
            
    def set_shio(self, shio_file):
        cdef ShioTimeValue_c *shio
        shio = new ShioTimeValue_c()
        if(shio.ReadTimeValues(shio_file) == -1):
            return False
        self.mover.SetTimeDep(shio)
        self.mover.SetRefPosition(shio.GetRefWorldPoint(), 0)
        self.mover.bTimeFileActive = True
        return True
        
    def set_ref_point(self, ref_point):
        cdef WorldPoint p
        p.pLong = ref_point[0]
        p.pLat = ref_point[1]
        self.mover.SetRefPosition(p, 0)
        
    def read_topology(self, path):
        cdef Map_c **naught
        #fixme: why might htis fail? 
        if(self.mover.ReadTopology(path, naught)):
            return False
        return True
            
    def get_move(self, int t, np.ndarray[LERec, ndim=1] LEs, uncertain, set_index):
        cdef int i    
        cdef WorldPoint3D wp3d
        cdef np.ndarray[LERec] ra = np.copy(LEs)
        cdef float dpLat, dpLong
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        uncertain += 1
        self.mover.PrepareForModelStep()
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, set_index, i, &ra[i], uncertain)
            dpLat = wp3d.p.pLat
            dpLong = wp3d.p.pLong
            LEs[i].p.pLat += (dpLat/1000000)
            LEs[i].p.pLong += (dpLong/1000000)
        self.mover.fOptimize.isOptimizedForStep = 1
        self.mover.fOptimize.isFirstStep = 0
        self.mover.ModelStepIsDone()
    
    def compute_velocity_scale(self):
        self.mover.ComputeVelocityScale()
        
    def set_velocity_scale(self, scale_value):
        self.mover.refScale = scale_value
        
cdef class random_mover:

    cdef Random_c *mover

    def __cinit__(self):
        self.mover = new Random_c()
        
    def __dealloc__(self):
        del self.mover
        
    def __init__(self, diffusion_coefficient):
        self.mover.bUseDepthDependent = 0                
        self.mover.fOptimize.isOptimizedForStep = 0
        self.mover.fOptimize.isFirstStep = 1           
        self.mover.fUncertaintyFactor = 2
        self.mover.fDiffusionCoefficient = diffusion_coefficient

    def get_move(self, int t, np.ndarray[LERec, ndim=1] LEs, uncertain, set_index):
        cdef int i    
        cdef WorldPoint3D wp3d
        cdef np.ndarray[LERec] ra = np.copy(LEs)
        cdef float dpLat, dpLong
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        uncertain += 1
        self.mover.PrepareForModelStep()
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &ra[i], uncertain)
            dpLat = wp3d.p.pLat
            dpLong = wp3d.p.pLong
            LEs[i].p.pLat += (dpLat/1000000)
            LEs[i].p.pLong += (dpLong/1000000)
        self.mover.ModelStepIsDone()


cdef class wind_mover:

    cdef WindMover_c *mover

    def __cinit__(self):
        self.mover = new WindMover_c()
        
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, constant_wind_value):
        """
        initialize a constant wind mover
        
        constant_wind_value is a tuple of values: (u, v)
        """
        self.mover.fUncertainStartTime = 0
        self.mover.fDuration = 3*3600                                
        self.mover.fSpeedScale = 2
        self.mover.fAngleScale = .4
        self.mover.fMaxSpeed = 30 #mps
        self.mover.fMaxAngle = 60 #degrees
        self.mover.fSigma2 = 0
        self.mover.fSigmaTheta = 0 
        self.mover.bUncertaintyPointOpen = 0
        self.mover.bSubsurfaceActive = 0
        self.mover.fGamma = 1
        self.mover.fIsConstantWind = 1
        self.mover.fConstantValue.u = constant_wind_value[0]
        self.mover.fConstantValue.v = constant_wind_value[1]
        self.mover.AllocateUncertainty()
        
    def get_move(self, t, np.ndarray[LERec, ndim=1] LEs, uncertain, set_index):
        cdef:
            int i
            WorldPoint3D wp3d
            float dpLat, dpLong
            np.ndarray[LERec] ra = np.copy(LEs)
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        uncertain += 1
        self.mover.PrepareForModelStep()
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, set_index, i, &ra[i], uncertain)
            dpLat = wp3d.p.pLat
            dpLong = wp3d.p.pLong
            LEs[i].p.pLat += (dpLat/1000000)
            LEs[i].p.pLong += (dpLong/1000000)

