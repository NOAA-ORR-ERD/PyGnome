cimport numpy as cnp
import numpy as np
import os

from type_defs cimport *
from movers cimport CATSMover_c,Mover_c
from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime
cimport cy_mover,cy_ossm_time

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition for each mover
"""
cdef extern from *:
    CATSMover_c* dynamic_cast_ptr "dynamic_cast<CATSMover_c *>" (Mover_c *) except NULL


cdef class CyCatsMover(cy_mover.CyMover):

    cdef CATSMover_c *cats
    
    def __cinit__(self):
        self.mover = new CATSMover_c()
        self.cats = dynamic_cast_ptr(self.mover)
    
    def __dealloc__(self):
        del self.mover  # since this is allocated in this class, free memory here as well
        self.cats = NULL
    
    def __init__(self, scale_type=0, scale_value=1, diffusion_coefficient=1):
        cdef WorldPoint p
        self.cats.scaleType = scale_type
        self.cats.scaleValue = scale_value
        self.cats.fEddyDiffusion = diffusion_coefficient
        ## should not have to do this manually.
        ## make-shifting for now.
        #self.cats.fOptimize.isOptimizedForStep = 0
        #self.cats.fOptimize.isFirstStep = 1


    property scale_type:
        def __get__(self):
            return self.cats.scaleType
        
        def __set__(self,value):
            self.cats.scaleType = value
    
    property scale_value:
        def __get__(self):
            return self.cats.scaleValue
        
        def __set__(self,value):
            self.cats.scaleValue = value
            
    property eddy_diffusion:
        def __get__(self):
            return self.cats.fEddyDiffusion
        
        def __set__(self,value):
            self.cats.fEddyDiffusion = value    

         
    def set_shio(self, CyShioTime cy_shio):
        self.cats.SetTimeDep(cy_shio.shio)
        self.cats.SetRefPosition(cy_shio.shio.GetRefWorldPoint(), 0)
        self.cats.bTimeFileActive = True
        self.cats.scaleType = 1
        return True
        
    def set_ossm(self, CyOSSMTime ossm):
        self.cats.SetTimeDep(ossm.time_dep)
        self.cats.bTimeFileActive = True   # What is this?
        return True
        
    def set_ref_point(self, ref_point):
        cdef WorldPoint p
        p.pLong = ref_point[0]*10**6    # should this happen in C++?
        p.pLat = ref_point[1]*10**6
        self.cats.SetRefPosition(p, 0)
            
    def read_topology(self, path):
        cdef OSErr err
        if os.path.exists(path):
            err = self.cats.ReadTopology(path)
            if err != False:
                raise ValueError("CATSMover.ReadTopology(..) returned an error. OSErr: {0}".format(err))
        else:
            raise IOError("No such file: " + path)
        
        return True
    

    def get_move(self, model_time, step_len, cnp.ndarray[WorldPoint3D, ndim=1] ref_points, cnp.ndarray[WorldPoint3D, ndim=1] delta, cnp.ndarray[cnp.npy_int16] LE_status, LEType spill_type, long spill_ID):
        cdef OSErr err
            
        N = len(ref_points)
 
        err = self.cats.get_move(N, model_time, step_len, &ref_points[0], &delta[0], <short *>&LE_status[0], spill_type, spill_ID)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points, delta and windages are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        
    
    #===========================================================================
    # TODO: What are these used for and make them into properties
    # def compute_velocity_scale(self, model_time):
    #    self.mover.ComputeVelocityScale(model_time)
    #    
    # def set_velocity_scale(self, scale_value):
    #    self.mover.refScale = scale_value
    #===========================================================================
        
