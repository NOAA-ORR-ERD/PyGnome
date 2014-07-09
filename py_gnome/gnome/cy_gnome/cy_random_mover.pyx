cimport numpy as cnp
import numpy as np

# following exist in gnome.cy_gnome
from type_defs cimport *
from movers cimport Random_c, Mover_c
cimport cy_mover

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition
for each mover
"""
cdef extern from *:
    Random_c* dynamic_cast_ptr "dynamic_cast<Random_c *>" (Mover_c *) except NULL

cdef class CyRandomMover(cy_mover.CyMover):

    cdef Random_c *rand

    def __cinit__(self):
        self.mover = new Random_c()
        self.rand = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        del self.mover
        self.rand = NULL

    def __init__(self, diffusion_coef=100000, uncertain_factor=2):
        """
        Default diffusion_coef = 100,000 [cm**2/sec]
        Default uncertain_factor = 2
        """
        if diffusion_coef < 0:
            raise ValueError('CyRandomMover must have a value '
                             'greater than or equal to 0 for diffusion_coef')
        if uncertain_factor < 1:
            raise ValueError('CyRandomMover must have a value '
                             'greater than or equal to 1 for uncertain_factor')

        self.rand.fDiffusionCoefficient = diffusion_coef
        self.rand.fUncertaintyFactor = uncertain_factor

    property diffusion_coef:
        def __get__(self):
            return self.rand.fDiffusionCoefficient

        def __set__(self, value):
            if value < 0:
                raise ValueError('CyRandomMover must have a value '
                                 'greater than or equal to 0 '
                                 'for diffusion_coef')
            self.rand.fDiffusionCoefficient = value

    property uncertain_factor:
        def __get__(self):
            return self.rand.fUncertaintyFactor

        def __set__(self, value):
            if value < 1:
                raise ValueError('CyRandomMover must have a value '
                                 'greater than or equal to 1 '
                                 'for uncertain_factor')
            self.rand.fUncertaintyFactor = value

    def __repr__(self):
        """
        unambiguous repr of object, reuse for str() method
        """
        return ('CyRandomMover(diffusion_coef={0}, uncertain_factor={1})'
                .format(self.diffusion_coef, self.uncertain_factor))

    def __reduce__(self):
        return (CyRandomMover, (self.diffusion_coef, self.uncertain_factor))

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
                 LE_type)

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
        N = len(ref_points)  # set a data type?

        err = self.rand.get_move(N, model_time, step_len, &ref_points[0], &delta[0], &LE_status[0], spill_type, 0)
        if err == 1:
            raise ValueError('Make sure numpy arrays for ref_points and delta '
                             'are defined')

        """
        Can probably raise this error before calling the C++ code
        - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError('The value for spill type can only be '
                             '"forecast" or "uncertainty" - you have chosen: '
                             '{0!s}'.format(spill_type))
