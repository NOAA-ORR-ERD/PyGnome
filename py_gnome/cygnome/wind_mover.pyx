import cython
cimport numpy as np
import numpy as nmp
include "wind_mover.pxi"

cdef class wind_mover:

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
    
    def get_move_uncertain(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] wp_ra, np.ndarray[np.npy_double] wind_ra, np.ndarray[np.npy_short] dispersion_ra, double breaking_wave, double mix_layer, np.ndarray[LEWindUncertainRec] uncertain_ra, np.ndarray[TimeValuePair] time_vals, int num_times):
        cdef:
            char *time_vals_ptr
            char *uncertain_ptr
            char *world_points
            char *windages
            char *disp_vals
            
        N = len(wp_ra)
        M = len(time_vals)

        wp_ra = nmp.require(wp_ra, requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'UPDATEIFCOPY'])
        print wp_ra.flags
        exit(1)
        world_points = np.PyArray_BYTES(wp_ra)
        windages = np.PyArray_BYTES(wind_ra)
        disp_vals = np.PyArray_BYTES(dispersion_ra)
        time_vals_ptr = np.PyArray_BYTES(time_vals)
        uncertain_ptr = np.PyArray_BYTES(uncertain_ra)
        
        self.mover.get_move(N, model_time, step_len, world_points, windages, disp_vals, breaking_wave, mix_layer, uncertain_ptr, time_vals_ptr, M)

    def get_move(self, n, model_time, step_len, np.ndarray[WorldPoint3D, ndim=1] wp_ra, np.ndarray[np.npy_double] wind_ra, np.ndarray[np.npy_short] dispersion_ra, double breaking_wave, double mix_layer, np.ndarray[TimeValuePair] time_vals, int num_times):
        cdef:
            char *time_vals_ptr
            char *uncertain_ptr
            char *world_points
            char *windages
            char *disp_vals
            
        N = len(wp_ra)
        M = len(time_vals)
        wp_ra = nmp.require(wp_ra, requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'UPDATEIFCOPY'])
        world_points = np.PyArray_BYTES(wp_ra)
        windages = np.PyArray_BYTES(wind_ra)
        disp_vals = np.PyArray_BYTES(dispersion_ra)
        time_vals_ptr = np.PyArray_BYTES(time_vals)
        self.mover.get_move(N, model_time, step_len, world_points, windages, disp_vals, breaking_wave, mix_layer, time_vals_ptr, M)
