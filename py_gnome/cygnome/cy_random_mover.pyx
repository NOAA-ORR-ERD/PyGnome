import cython
DEF HEADERS = list()
cimport numpy as np
import numpy as np
include "type_defs.pxi"
include "random_mover.pxi"

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

