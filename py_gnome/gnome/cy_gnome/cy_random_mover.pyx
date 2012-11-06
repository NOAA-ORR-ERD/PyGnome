import cython
cimport numpy as np
import numpy as np

from type_defs cimport *
from movers cimport Random_c

cdef class CyRandomMover:

    cdef Random_c *mover

    def __cinit__(self):
        self.mover = new Random_c()
        
    def __dealloc__(self):
        del self.mover
        
    def __init__(self, diffusion_coef=100000):
        """
        Default diffusion_coef = 100,000 [cm**2/sec]
        """
        if diffusion_coef <= 0:
            raise ValueError("CyRandomMover must have a value greater than or equal to 0 for diffusion_coef")
        
        self.mover.fDiffusionCoefficient = diffusion_coef
    
    property diffusion_coef:
        def __get__(self):
            return self.mover.fDiffusionCoefficient
        
        def __set__(self, value):
            self.mover.fDiffusionCoefficient = value
        
    def prepare_for_model_run(self):
        """
        .. function::prepare_for_model_run
        
        """
        self.mover.PrepareForModelRun()
        
    def prepare_for_model_step(self, model_time, step_len, uncertain=False):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: current model time. Not used by Random_mover.
        :param step_len: length of the time step over which the get move will be computed
        :param uncertain: bool flag determines whether to apply uncertainty or not - again, not used by mover
        """
        cdef OSErr err
        err = self.mover.PrepareForModelStep(model_time, step_len, uncertain, 0, NULL)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("Random_c.PreareForModelStep returned an error.")

    def get_move(self, 
                 model_time, 
                 step_len, 
                 np.ndarray[WorldPoint3D, ndim=1] ref_points, 
                 np.ndarray[WorldPoint3D, ndim=1] delta, 
                 np.ndarray[np.npy_int16] LE_status, 
                 LEType spill_type, 
                 long spill_ID):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type,
                 spill_ID)
                 
        Invokes the underlying C++ Random_c.get_move(...)
        
        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param le_status: status of each particle - movement is only on particles in water
        :param spill_type: LEType defining whether spill is forecast or uncertain 
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points) # set a data type?
            
        err = self.mover.get_move(N, model_time, step_len, &ref_points[0], &delta[0], <short *>&LE_status[0], spill_type, spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta and windages are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        

        # JS Ques: Should we call this from within get_move instead of calling it from cython?
        self.mover.ModelStepIsDone()
