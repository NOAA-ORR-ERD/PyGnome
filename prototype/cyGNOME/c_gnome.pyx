import cython
import random
import math

from libcpp.vector cimport vector
from cython.operator import preincrement as preinc

cimport numpy as np
import numpy as np

include "c_gnome_defs.pxi"
    
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

    def get_move(self, int t, np.ndarray[LERec, ndim=1] LEs):
        cdef int i    
        cdef WorldPoint3D wp3d
        
        for i in xrange(0, len(LEs)):
            if LEs[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &LEs[i], 0)          
            LEs[i].p = wp3d.p
            LEs[i].z = wp3d.z

cdef class wind_mover:

    cdef WindMover_c *mover

    def __cinit__(self):
        self.mover = new WindMover_c()
        
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, constant_wind_value):
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

    def get_move(self, t, np.ndarray[LERec, ndim=1] LEs):
        
        cdef int i
        cdef WorldPoint3D wp3d
        
        for i in xrange(0, len(LEs)):
            if LEs[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, 0, 0, &LEs[i], 0)
            LEs[i].p = wp3d.p
            LEs[i].z = wp3d.z        
