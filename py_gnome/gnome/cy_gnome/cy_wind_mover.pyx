import cython
from cython.operator cimport dereference as deref
cimport numpy as np
import numpy as nmp

from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from movers cimport WindMover_c
from type_defs cimport WorldPoint3D, LEWindUncertainRec, LEStatus, LEType, OSErr

cdef class CyWindMover:

    cdef WindMover_c *mover

    def __cinit__(self):
        self.mover = new WindMover_c()
        
    def __dealloc__(self):
        del self.mover
    
    def __init__(self):
        """
        initialize a constant wind mover
        
        constant_wind_value is a tuple of values: (u, v)
        """
        self.mover.fIsConstantWind  = 0  # don't assume wind is constant
        self.mover.fConstantValue.u = 0
        self.mover.fConstantValue.v = 0
    

    def get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status,
                 LEType spill_type):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type)
                 
        The cython wind mover's private get move method. It invokes the underlying C++ WindMover_c.get_move(...)
        
        :param model_time: 
        :param step_len:
        :param ref_points:
        :type ref_points:        
        :param delta:
        :type delta:
        :param LE_windage:
        :type LE_windage:
        :param le_status:
        :param LE_type:
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
                                  <LEStatus *>&LE_status[0],
                                  spill_type)
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
       
