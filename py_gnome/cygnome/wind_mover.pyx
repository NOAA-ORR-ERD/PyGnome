import cython
DEF HEADERS = list()
cimport numpy as np
import numpy as np
include "type_defs.pxi"
include "wind_mover.pxi"

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
        
    def get_move(self, t, np.ndarray[LERec, ndim=1] LEs, uncertain, set_index):
        cdef:
            int i
            WorldPoint3D wp3d
            float dpLat, dpLong
            np.ndarray[LERec] ra = np.copy(LEs)
        ra['p']['p_long']*=10**6
        ra['p']['p_lat']*=10**6
        uncertain += 1
        self.mover.PrepareForModelStep()
        for i in xrange(0, len(ra)):
            if ra[i].statusCode != status_in_water:
                continue
            wp3d = self.mover.GetMove(t, set_index, i, &ra[i], uncertain)
            dpLat = wp3d.p.pLat
            dpLong = wp3d.p.pLong
            LEs[i].p.pLat += (dpLat/1000000)
            LEs[i].p.pLong += (dpLong/1000000)

