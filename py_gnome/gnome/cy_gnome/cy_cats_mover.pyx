import cython
cimport numpy as cnp
import numpy as np

from type_defs cimport *

from movers cimport CATSMover_c
from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime

cdef class CyCatsMover:

    cdef CATSMover_c *mover
    
    def __cinit__(self):
        self.mover = new CATSMover_c()
    
    def __dealloc__(self):
        del self.mover
    
    def __init__(self, scale_type=0, scale_value=1, diffusion_coefficient=1):
        cdef WorldPoint p
        self.mover.scaleType = scale_type
        self.mover.scaleValue = scale_value
        self.mover.fEddyDiffusion = diffusion_coefficient
        ## should not have to do this manually.
        ## make-shifting for now.
        #self.mover.fOptimize.isOptimizedForStep = 0
        #self.mover.fOptimize.isFirstStep = 1


    property scale_type:
        def __get__(self):
            return self.mover.scaleType
        
        def __set__(self,value):
            self.mover.scaleType = value
    
    property scale_value:
        def __get__(self):
            return self.mover.scaleValue
        
        def __set__(self,value):
            self.mover.scaleValue = value
            
    property eddy_diffusion:
        def __get__(self):
            return self.mover.fEddyDiffusion
        
        def __set__(self,value):
            self.mover.fEddyDiffusion = value    

         
    def set_shio(self, CyShioTime cy_shio):
        self.mover.SetTimeDep(cy_shio.shio)
        self.mover.SetRefPosition(cy_shio.shio.GetRefWorldPoint(), 0)
        self.mover.bTimeFileActive = True
        return True
        
    def set_ossm(self, CyOSSMTime ossm):
        self.mover.SetTimeDep(ossm.time_dep)
        self.mover.bTimeFileActive = True   # What is this?
        return True
        
    def set_ref_point(self, ref_point):
        cdef WorldPoint p
        p.pLong = ref_point[0]*10**6    # should this happen in C++?
        p.pLat = ref_point[1]*10**6
        self.mover.SetRefPosition(p, 0)
            
    #===========================================================================
    # TODO: don't have map - is this required?
    # def read_topology(self, path):
    #    cdef Map_c **naught
    #    if(self.mover.ReadTopology(path, naught)):
    #        return False
    #    return True
    #===========================================================================
    
    def prepare_for_model_run(self):
        """
        .. function::prepare_for_model_run
        
        """
        self.mover.PrepareForModelRun()

        
    def prepare_for_model_step(self, model_time, step_len, numSets=0, cnp.ndarray[cnp.npy_int] setSizes=None):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: current model time.
        :param step_len: length of the time step over which the get move will be computed
        """
        cdef OSErr err
        if numSets == 0:
            err = self.mover.PrepareForModelStep(model_time, step_len, False, 0, NULL)
        else:
            err = self.mover.PrepareForModelStep(model_time, step_len, True, numSets, <int *>&setSizes[0])
            
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("WindMover_c.PreareForModelStep returned an error.")
        
    def model_step_is_done(self):
        """
        invoke C++ model step is done functionality
        """
        self.mover.ModelStepIsDone()

    def get_move(self, model_time, step_len, cnp.ndarray[WorldPoint3D, ndim=1] ref_points, cnp.ndarray[WorldPoint3D, ndim=1] delta, cnp.ndarray[cnp.npy_int16] LE_status, LEType spill_type, long spill_ID):
        cdef OSErr err
            
        N = len(ref_points)
 
        err = self.mover.get_move(N, model_time, step_len, &ref_points[0], &delta[0], <short *>&LE_status[0], spill_type, spill_ID)
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
        
