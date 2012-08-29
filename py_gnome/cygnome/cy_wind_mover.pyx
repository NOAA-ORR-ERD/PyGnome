import cython
cimport numpy as np
import numpy as nmp

include "wind_mover.pxi"

cdef class Cy_wind_mover:

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
        self.mover.breaking_wave_height = 0
        # AH 06/20/2012: should there be a default value here?
        # self.mover.mix_layer_depth = ??     
        
        self.mover.fUncertainStartTime = 0
        self.mover.fDuration = 3*3600                                
        self.mover.fSpeedScale = 2
        self.mover.fAngleScale = .4
        self.mover.fMaxSpeed = 30 #mps
        self.mover.fMaxAngle = 60 #degrees
        self.mover.fSigma2 = 0
        self.mover.fSigmaTheta = 0 
        self.mover.bUncertaintyPointOpen = 0
        self.mover.bSubsurfaceActive = 0
        self.mover.fGamma = 1
        self.mover.fIsConstantWind = 1
        self.mover.fConstantValue.u = 0
        self.mover.fConstantValue.v = 0
    
    def get_move_uncertain(self,
                           model_time,
                           step_len,
                           np.ndarray[WorldPoint3D, ndim=1] ref_ra,
                           np.ndarray[WorldPoint3D, ndim=1] wp_ra,
                           np.ndarray[np.npy_double] wind_ra,
                           np.ndarray[np.npy_short] dispersion_ra,
                           double f_sigma_vel,
                           double f_sigma_theta,
                           double breaking_wave,
                           double mix_layer,
                           np.ndarray[LEWindUncertainRec] uncertain_ra,
                           np.ndarray[TimeValuePair] time_vals):
        cdef:
            char *time_vals_ptr
            char *uncertain_ptr
            char *delta
            char *windages
            char *disp_vals
            
        N = len(wp_ra) # set a data type here?
        M = len(time_vals) # set a data type here?
        ref_points = ref_ra.data
        delta = wp_ra.data
        windages = wind_ra.data
        disp_vals = dispersion_ra.data
        time_vals_ptr = time_vals.data
        uncertain_ptr = uncertain_ra.data
        
        self.mover.get_move(N, model_time, step_len, ref_points, delta, windages, disp_vals, f_sigma_vel, f_sigma_theta, breaking_wave, mix_layer, uncertain_ptr, time_vals_ptr, M)

    ##fixme: don't need breaking wave, etc...
    ## need to clarify what is going on here -- is the delta put into the wp_ra??    
    def get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_short] dispersion_flags,
                 double breaking_wave,
                 double mix_layer,
                 np.ndarray[TimeValuePair] time_vals):

        N = len(ref_points) # set a data type?
        M = len(time_vals) # set a data type?

        # modifies delta in place
        self.mover.get_move(N,
                            model_time,
                            step_len,
                            <char*>&ref_points[0],
                            <char*>&delta[0],
                            <char*>&windages[0],
                            <char*>&dispersion_flags[0],
                            breaking_wave, # remove?
                            mix_layer, # remove?
                            <char*>&time_vals[0], # remove?
                            M,# remove?
                            )
