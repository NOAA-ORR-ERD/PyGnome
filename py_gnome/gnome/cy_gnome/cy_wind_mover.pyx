import cython
from cython.operator cimport dereference as deref
cimport numpy as np
import numpy as nmp

from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from movers cimport WindMover_c
from type_defs cimport WorldPoint3D, LEWindUncertainRec

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
    
    def get_move_uncertain(self,
                           model_time,
                           step_len,
                           np.ndarray[WorldPoint3D, ndim=1] ref_ra,
                           np.ndarray[WorldPoint3D, ndim=1] wp_ra,
                           np.ndarray[np.npy_double] windage_ra,
                           double f_sigma_vel,
                           double f_sigma_theta,
                           np.ndarray[LEWindUncertainRec] uncertain_ra):
        cdef:
            char *uncertain_ptr
            char *delta
            char *windages
            
        N = len(wp_ra) # set a data type here?
        ref_points = ref_ra.data
        delta = wp_ra.data
        windages = windage_ra.data
        uncertain_ptr = uncertain_ra.data
        
        self.mover.get_move(N, model_time, step_len, ref_points, delta, windages, f_sigma_vel, f_sigma_theta, uncertain_ptr)

    def get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages):

        N = len(ref_points) # set a data type?

        # modifies delta in place
        self.mover.get_move(N,
                             model_time,
                             step_len,
                             &ref_points[0],
                             &delta[0],
                             &windages[0])
        

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
        
       