import cython
cimport numpy as np
import numpy as np
include "type_defs.pxi"
include "random_mover.pxi"

cdef class Cy_random_mover:

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

    #for now there is no difference between the two functions    
    def get_move_uncertain(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra):
        cdef:
            char *world_points
         
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data

        self.mover.get_move(N, model_time, step_len, ref_points, world_points)

    def get_move(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra):
        cdef:
            char *world_points
            
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data

        self.mover.get_move(N, model_time, step_len, ref_points, world_points)

