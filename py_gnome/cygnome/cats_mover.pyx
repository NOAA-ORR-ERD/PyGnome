import cython
DEF HEADERS = list()
cimport numpy as np
import numpy as np
include "type_defs.pxi"
include "map.pxi"
include "cats_mover.pxi"
include "shio_time.pxi"


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
        p.pLong = ref_point[0]*10**6
        p.pLat = ref_point[1]*10**6
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
        
