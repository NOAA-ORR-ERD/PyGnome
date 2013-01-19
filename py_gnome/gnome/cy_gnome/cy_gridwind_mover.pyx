cimport numpy as np
import numpy as np

from type_defs cimport *
from movers cimport Mover_c,GridWindMover_c,TimeGridVel_c,TimeGridVelRect_c,TimeGridVelCurv_c,TimeGridWindRect_c,TimeGridWindCurv_c
cimport cy_mover
"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition for each mover
"""
cdef extern from *:
    GridWindMover_c* dynamic_cast_ptr "dynamic_cast<GridWindMover_c *>" (Mover_c *) except NULL
    
cdef class CyGridWindMover(cy_mover.CyMover):

    cdef GridWindMover_c *grid
    
    def __cinit__(self):
        self.mover = new GridWindMover_c()
        self.grid = dynamic_cast_ptr(self.mover)
    
    def __dealloc__(self):
        del self.mover
        self.grid = NULL
    
#     def set_time_grid(self, time_grid_file, topology_file):
#         self.grid.fIsOptimizedForStep = 0
#         #cdef TimeGridVel_c *time_grid
#         cdef TimeGridVelCurv_c *time_grid
#         time_grid = new TimeGridVelCurv_c()
#         #time_grid = new TimeGridVel_c()
#         if (time_grid.TextRead(time_grid_file, topology_file) == -1):
#             return False
#         self.grid.SetTimeGrid(time_grid)
#         return True
            

    def text_read(self, time_grid_file, topology_file):
        """
        .. function::text_read
        
        """
        cdef OSErr err
        err = self.grid.TextRead(time_grid_file, topology_file)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridWindMover_c.TextRead returned an error.")

    def __init__(self):
        self.grid.fIsOptimizedForStep = 0

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

        err = self.grid.get_move(N, model_time, step_len, &ref_points[0], &delta[0], &windages[0], <short *>&LE_status[0], spill_type, spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points and delta are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
