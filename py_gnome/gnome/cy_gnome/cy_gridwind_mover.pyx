import cython
cimport numpy as np
import numpy as np

from type_defs cimport *
from movers cimport GridWindMover_c
from movers cimport TimeGridVel_c
from movers cimport TimeGridVelRect_c
from movers cimport TimeGridVelCurv_c
from movers cimport TimeGridWindRect_c
from movers cimport TimeGridWindCurv_c

cdef class CyGridWindMover:

    cdef GridWindMover_c *mover
    
    def __cinit__(self):
        self.mover = new GridWindMover_c()
    
    def __dealloc__(self):
        del self.mover
    
#     def set_time_grid(self, time_grid_file, topology_file):
#         self.mover.fIsOptimizedForStep = 0
#         #cdef TimeGridVel_c *time_grid
#         cdef TimeGridVelCurv_c *time_grid
#         time_grid = new TimeGridVelCurv_c()
#         #time_grid = new TimeGridVel_c()
#         if (time_grid.TextRead(time_grid_file, topology_file) == -1):
#             return False
#         self.mover.SetTimeGrid(time_grid)
#         return True
            

    def text_read(self, time_grid_file, topology_file):
        """
        .. function::text_read
        
        """
        cdef OSErr err
        err = self.mover.TextRead(time_grid_file, topology_file)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridWindMover_c.TextRead returned an error.")

    def __init__(self):
        self.mover.fIsOptimizedForStep = 0

    def prepare_for_model_run(self):
        """
        .. function::prepare_for_model_run
        
        """
        self.mover.PrepareForModelRun()
        
    def prepare_for_model_step(self, model_time, step_len, numSets=0, np.ndarray[np.npy_int] setSizes=None):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: current model time.
        :param step_len: length of the time step over which the get move will be computed
        """
        cdef OSErr err
        if numSets == 0:
            err = self.mover.PrepareForModelStep(model_time, step_len, False, 0, NULL)
        else:
            err = self.mover.PrepareForModelStep(model_time, step_len, True, numSets, <int *>&setSizes[0])
            
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridWindMover_c.PrepareForModelStep returned an error.")

    def get_move(self, 
                 model_time, 
                 step_len, 
                 np.ndarray[WorldPoint3D, ndim=1] ref_points, 
                 np.ndarray[WorldPoint3D, ndim=1] delta, 
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status, 
                 LEType spill_type, 
                 long spill_ID):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type,
                 spill_ID)
                 
        Invokes the underlying C++ GridWindMover_c.get_move(...)
        
        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param LE_windage: windage to be applied to each particle
        :type LE_windage: numpy array of numpy.npy_int16
        :param le_status: status of each particle - movement is only on particles in water
        :param spill_type: LEType defining whether spill is forecast or uncertain 
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points) 

        err = self.mover.get_move(N, model_time, step_len, &ref_points[0], &delta[0], &windages[0], <short *>&LE_status[0], spill_type, spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points and delta are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        

    def model_step_is_done(self):
        """
        invoke C++ model step is done functionality
        """
        self.mover.ModelStepIsDone()
