import cython
cimport numpy as np
import numpy as np

include "netcdf_mover.pxi"

cdef class netcdf_mover:

    cdef NetCDFMover_c *mover
    
    def __cinit__(self):
        self.mover = new NetCDFMover_c()
    
    def __dealloc__(self):
        del self.mover
    
    def __init__(self):

        self.mover.isOptimizedForStep = 0

    def get_move_uncertain(self, n, start_time, stop_time, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra, np.ndarray[LEWindUncertainRec] uncertain_ra):
        cdef:
            char *time_vals_ptr
            char *uncertain_ptr
            char *world_points
            
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data
        uncertain_ptr = uncertain_ra.data

        self.mover.get_move(N, start_time, stop_time, model_time, step_len, ref_points, world_points, uncertain_ptr)

    def get_move(self, n, start_time, stop_time, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_ra, np.ndarray[WorldPoint3D, ndim=1] wp_ra):
        cdef:
            char *time_vals_ptr
            char *uncertain_ptr
            char *world_points
            
        N = len(wp_ra)
        ref_points = ref_ra.data
        world_points = wp_ra.data

        self.mover.get_move(N, start_time, stop_time, model_time, step_len, ref_points, world_points)
