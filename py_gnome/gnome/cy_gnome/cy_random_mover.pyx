import cython
cimport numpy as np
import numpy as np

from type_defs cimport *
from movers cimport Random_c

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

    def get_move(self, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_points, np.ndarray[WorldPoint3D, ndim=1] delta, np.ndarray[np.npy_int16] LE_status, LEType spill_type):
        cdef OSErr err
        N = len(ref_points) # set a data type?
            
        err = self.mover.get_move(N, model_time, step_len, &ref_points[0], &delta[0], <short *>&LE_status[0], spill_type)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta and windages are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        

