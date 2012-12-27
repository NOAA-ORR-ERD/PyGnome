import cython
cimport numpy as np
import numpy as np

from type_defs cimport *

from movers cimport CATSMover_c,Map_c
from utils cimport OSSMTimeValue_c,ShioTimeValue_c


cdef class CyCatsMover:

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
        #self.mover.fOptimize.isOptimizedForStep = 0
        #self.mover.fOptimize.isFirstStep = 1
            
    def set_shio(self, shio_file):
        cdef ShioTimeValue_c *shio
        shio = new ShioTimeValue_c()
        if(shio.ReadTimeValues(shio_file) == -1):
            return False
        self.mover.SetTimeDep(shio)
        self.mover.SetRefPosition(shio.GetRefWorldPoint(), 0)
        self.mover.bTimeFileActive = True
        return True
        
    def set_ossm(self, CyOSSMTime ossm):
        self.mover.SetTimeDep(ossm.time_dep)
        self.mover.bTimeFileActive = True   # What is this?
        return True
        
    def set_ref_point(self, ref_point):
        cdef WorldPoint p
        p.pLong = ref_point[0]*10**6    # should this happen in C++?
        p.pLat = ref_point[1]*10**6
        self.mover.SetRefPosition(p, 0)
        
    def read_topology(self, path):
        cdef Map_c **naught
        if(self.mover.ReadTopology(path, naught)):
            return False
        return True
    
    def get_move(self, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] ref_points, np.ndarray[WorldPoint3D, ndim=1] delta, np.ndarray[np.npy_int16] LE_status, LEType spill_type, long spill_ID):
        cdef OSErr err
            
        N = len(ref_points)

        err = self.mover.get_move(N, model_time, step_len, &ref_points[0], &delta[0], <short *>&LE_status[0], spill_type, spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta and windages are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        

    def compute_velocity_scale(self, model_time):
        self.mover.ComputeVelocityScale(model_time)
        
    def set_velocity_scale(self, scale_value):
        self.mover.refScale = scale_value
        
