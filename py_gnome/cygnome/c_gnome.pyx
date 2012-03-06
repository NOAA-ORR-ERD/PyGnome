import cython
import random
import math

from libcpp.vector cimport vector
from cython.operator import preincrement as preinc

cimport numpy as np
import numpy as np

include "c_gnome_defs.pxi"

#======================================================================#
# cdef class shio_time_value:
#     cdef ShioTimeValue_c *time_value
#     
#     def __cinit__(self):
#         self.time_value = new ShioTimeValue_c()
#         
#     def __dealloc__(self):
#         del self.time_value
#         
#     def __init__(self):
#        pass
#     
#     def read_time_values(self, path, format, units):
#         self.time_value.ReadTimeValues(path, format, units)
#
#=====================================================================#
        
cdef class cats_mover:

    cdef CATSMover_c *mover
    
    def __cinit__(self):
        self.mover = new CATSMover_c()
    
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, scale_type, scale_value=1, diffusion_coefficient=1, shio_file=None, start_time=None, stop_time=None):
        cdef ShioTimeValue_c *shio
        self.mover.scaleType = scale_type
        self.mover.scaleValue = scale_value
        self.mover.fEddyDiffusion = diffusion_coefficient
        ## should not have to do this manually.
        ## make-shifting for now.
        self.mover.fOptimize.isOptimizedForStep = 0
        self.mover.fOptimize.isFirstStep = 1  
        if not shio_file or not start_time or not stop_time:
            pass
        else:
            shio = new ShioTimeValue_c(start_time, stop_time)
            shio.ReadTimeValues(shio_file, 0, 0)
            self.mover.SetTimeDep(shio)

    def read_topology(self, path):
        cdef Map_c **naught
        self.mover.ReadTopology(path, naught)
        
    def get_move(self, int t, np.ndarray[LERec, ndim=1] LEs, Seconds model_time):
        cdef int i    
        cdef WorldPoint3D wp3d
        cdef np.ndarray[LERec] ra = np.copy(LEs)
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &ra[i], 0, model_time)
            LEs[i].p.pLat += (wp3d.p.pLat)
            LEs[i].p.pLong += wp3d.p.pLong
            
    def set_ref_position(self, wp, z):
        cdef WorldPoint p
        p.pLong = wp[0]*10**6
        p.pLat = wp[1]*10**6
        self.mover.SetRefPosition(p, z)
    
    def compute_velocity_scale(self):
        self.mover.ComputeVelocityScale()
        
cdef class random_mover:

    cdef Random_c *mover

    def __cinit__(self):
        self.mover = new Random_c()
        
    def __dealloc__(self):
        del self.mover
        
    def __init__(self, diffusion_coefficient):
        self.mover.bUseDepthDependent = 0                
        self.mover.fOptimize.isOptimizedForStep = 0
        self.mover.fOptimize.isFirstStep = 1           
        self.mover.fUncertaintyFactor = 2
        self.mover.fDiffusionCoefficient = diffusion_coefficient

    def get_move(self, int t, np.ndarray[LERec, ndim=1] LEs, Seconds model_time = 0):
        cdef int i    
        cdef WorldPoint3D wp3d
        cdef np.ndarray[LERec] ra = np.copy(LEs)
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &ra[i], 0)
            LEs[i].p.pLat += (wp3d.p.pLat)
            LEs[i].p.pLong += wp3d.p.pLong

cdef class wind_mover:

    cdef WindMover_c *mover

    def __cinit__(self):
        self.mover = new WindMover_c()
        
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, constant_wind_value):
        """
        initialize a constant wind mover
        
        constant_wind_value is a tuple of values: (u, v)
        """
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
        self.mover.fConstantValue.u = constant_wind_value[0]
        self.mover.fConstantValue.v = constant_wind_value[1]

    def get_move(self, t, np.ndarray[LERec, ndim=1] LEs, Seconds model_time = 0):
        
        cdef int i
        cdef WorldPoint3D wp3d
        cdef np.ndarray[LERec] ra = np.copy(LEs)
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &ra[i], 0)
            LEs[i].p.pLat += wp3d.p.pLat
            LEs[i].p.pLong += wp3d.p.pLong
