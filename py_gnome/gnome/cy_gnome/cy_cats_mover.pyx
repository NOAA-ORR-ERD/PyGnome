import cython
cimport numpy as np
import numpy as np

from type_defs cimport *

from movers cimport CATSMover_c,Map_c
from utils cimport OSSMTimeValue_c,ShioTimeValue_c


cdef class Cy_cats_mover:

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
        
    def set_ossm(self, ossm_file):
        cdef OSSMTimeValue_c *ossm
        ossm = new OSSMTimeValue_c()
        if(ossm.ReadTimeValues(ossm_file,1,1) == -1):
            return False
        self.mover.SetTimeDep(ossm)
        #self.mover.SetRefPosition(ossm.GetRefWorldPoint(), 0)
        self.mover.bTimeFileActive = True
        return True
        
    def set_ref_point(self, ref_point):
        cdef WorldPoint p
        p.pLong = ref_point[0]*10**6
        p.pLat = ref_point[1]*10**6
        self.mover.SetRefPosition(p, 0)
        
    def read_topology(self, path):
        cdef Map_c **naught
        if(self.mover.ReadTopology(path, naught)):
            return False
        return True
    
    def get_move_uncertain(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra, np.ndarray[LEWindUncertainRec] uncertain_ra):
        cdef:
            char *uncertain_ptr
            char *world_points
            
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data
        uncertain_ptr = uncertain_ra.data

        self.mover.get_move(N, model_time, step_len, ref_points, world_points, uncertain_ptr)

    def get_move(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra):
        cdef:
            char *uncertain_ptr
            char *world_points
            
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data

        self.mover.get_move(N, model_time, step_len, ref_points, world_points)

    def compute_velocity_scale(self, model_time):
        self.mover.ComputeVelocityScale(model_time)
        
    def set_velocity_scale(self, scale_value):
        self.mover.refScale = scale_value
        
