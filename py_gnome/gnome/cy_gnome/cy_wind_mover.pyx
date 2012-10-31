import cython
from cython.operator cimport dereference as deref
cimport numpy as np
import numpy as nmp

from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome import basic_types

from movers cimport WindMover_c
from type_defs cimport WorldPoint3D, LEWindUncertainRec, LEStatus, LEType, OSErr

cdef class CyWindMover:

    cdef WindMover_c *mover

    def __cinit__(self):
        self.mover = new WindMover_c()
        
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, uncertain_duration=10800, 
                 uncertain_speed_scale=2, uncertain_max_speed=30, 
                 uncertain_angle_scale=0.4, uncertain_max_angle=60):
        """
        initialize a constant wind mover
        
        constant_wind_value is a tuple of values: (u, v)
        """
        self.mover.fIsConstantWind  = 0  # don't assume wind is constant
        self.mover.fConstantValue.u = 0
        self.mover.fConstantValue.v = 0
        self.mover.fDuration = uncertain_duration
        self.mover.fSpeedScale = uncertain_speed_scale
        self.mover.fMaxSpeed = uncertain_max_speed
        self.mover.fAngleScale = uncertain_angle_scale
        self.mover.fMaxAngle = uncertain_max_angle
    

    def prepare_for_model_step(self, model_time, step_len, uncertain):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: current model time.
        :param step_len: length of the time step over which the get move will be computed
        :param uncertain: bool flag determines whether to apply uncertainty or not
        """
        cdef OSErr err
        err = self.mover.PrepareForModelStep(model_time, step_len, uncertain, 0, NULL)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("WindMover_c.PreareForModelStep returned an error.")

    def prepare_for_model_step_uncertain(self, model_time, step_len, uncertain, numSets, np.ndarray[np.npy_int] setSizes):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: 
        :param step_len:
        :param uncertain: bool flag determines whether to apply uncertainty or not
        """
        cdef OSErr err
        numSets = len(setSizes) 

        err = self.mover.PrepareForModelStep(model_time, step_len, uncertain, numSets, <int *>&setSizes[0])
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("WindMover_c.PreareForModelStep returned an error.")

    def get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status,    # TODO: would be nice if we could define this as LEStatus type
                 LEType spill_type,
                 spill_ID):
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
                 
        Invokes the underlying C++ WindMover_c.get_move(...)
        
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
        N = len(ref_points) # set a data type?
        
        # modifies delta in place
        err = self.mover.get_move(N,
                                  model_time,
                                  step_len,
                                  &ref_points[0],
                                  &delta[0],
                                  &windages[0],
                                  <short *>&LE_status[0],
                                  spill_type,
                                  spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta and windages are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        

    def set_constant_wind(self,windU,windV):
    
        self.mover.fConstantValue.u = windU
        self.mover.fConstantValue.v = windV
        self.mover.fIsConstantWind = 1
        
    def set_ossm(self, CyOSSMTime ossm):
        """
        Use the CyOSSMTime object to set the wind mover OSSM time member variable using
        the SetTimeDep method
        """
        self.mover.SetTimeDep(ossm.time_dep)
        self.mover.fIsConstantWind = 0
        return True
       
