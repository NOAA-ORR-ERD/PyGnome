cimport numpy as cnp
import numpy as np

# following exist in gnome.cy_gnome 
from type_defs cimport *
from movers cimport RandomVertical_c,Mover_c
cimport cy_mover

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition for each mover
"""
cdef extern from *:
    RandomVertical_c* dynamic_cast_ptr "dynamic_cast<RandomVertical_c *>" (Mover_c *) except NULL

cdef class CyRandomVerticalMover(cy_mover.CyMover):

    cdef RandomVertical_c *rand

    def __cinit__(self):
        self.mover = new RandomVertical_c()
        self.rand = dynamic_cast_ptr(self.mover)
        
    def __dealloc__(self):
        del self.mover
        self.rand = NULL
        
    def __init__(self, vertical_diffusion_coef_above_ml=5, vertical_diffusion_coef_below_ml=.11):
        """
        Default vertical_diffusion_coef_above_ml = 5 [cm**2/sec]
        Default vertical_diffusion_coef_below_ml = .11 [cm**2/sec]
        """
        if vertical_diffusion_coef_above_ml < 0:
            raise ValueError("CyRandomVerticalMover must have a value greater than or equal to 0 for vertical_diffusion_coef above mixed layer")
        
        if vertical_diffusion_coef_below_ml < 0:
            raise ValueError("CyRandomVerticalMover must have a value greater than or equal to 0 for vertical_diffusion_coef below mixed layer")

        self.rand.fVerticalDiffusionCoefficient = vertical_diffusion_coef_above_ml
        self.rand.fVerticalBottomDiffusionCoefficient = vertical_diffusion_coef_below_ml
    
    property vertical_diffusion_coef_above_ml:
        def __get__(self):
            return self.rand.fVerticalDiffusionCoefficient
        
        def __set__(self, value):
            if value < 0:
                raise ValueError("CyRandomVerticalMover must have a value greater than or equal to 0 for vertical_diffusion_coef above mixed layer")
            self.rand.fVerticalDiffusionCoefficient = value

    property vertical_diffusion_coef_below_ml:
        def __get__(self):
            return self.rand.fVerticalBottomDiffusionCoefficient
        
        def __set__(self, value):
            if value < 0:
                raise ValueError("CyRandomVerticalMover must have a value greater than or equal to 0 for vertical_diffusion_coef below mixed layer")
            self.rand.fVerticalBottomDiffusionCoefficient = value

    def __repr__(self):
        """
        unambiguous repr of object, reuse for str() method
        """
        return "CyRandomVerticalMover(vertical_diffusion_coef=%s)" % self.vertical_diffusion_coef_above_ml
    
    def get_move(self, 
                 model_time, 
                 step_len, 
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points, 
                 cnp.ndarray[WorldPoint3D, ndim=1] delta, 
                 cnp.ndarray[short] LE_status, 
                 LEType spill_type):
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
            
        err = self.rand.get_move(N, model_time, step_len, &ref_points[0], &delta[0], &LE_status[0], spill_type, 0)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        
