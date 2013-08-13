import os

cimport numpy as cnp
import numpy as np

from type_defs cimport *
from movers cimport CATSMover_c,Mover_c
from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime
from gnome.cy_gnome cimport cy_mover
from gnome.cy_gnome.cy_helpers cimport to_bytes

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
    
    def __init__(self, scale_type=0, scale_value=1, diffusion_coefficient=0):
        cdef WorldPoint p
        self.cats.scaleType = scale_type
        self.cats.scaleValue = scale_value
        self.cats.fEddyDiffusion = diffusion_coefficient
        self.cats.refZ = -999 # default to -1
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
    
    property ref_point:
        def __get__(self):
            """
            returns the tuple containing (long, lat, z) of reference point if it is defined
            either by the user or obtained from the Shio object; otherwise it returns None 
            
            todo: make sure this is consistent with the format of CyShioTime.ref_point
            """
            if self.cats.refZ == -999:
                return None
            else:
                return (self.cats.refP.pLong/1.e6, self.cats.refP.pLat/1.e6, self.cats.refZ)
    
        def __set__(self,ref_point):
            """
            accepts a list or a tuple
            will not work with a numpy array since indexing assumes a list or a tuple
            
            takes only (long, lat, z), if length is bigger than 3, it uses the 1st 3 datapoints
            
            todo: make sure this is consistent with the format of CyShioTime.ref_point 
            """
            ref_point = np.asarray(ref_point)   # make it a numpy array
            cdef WorldPoint p
            p.pLong = ref_point[0]*10**6    # should this happen in C++?
            p.pLat = ref_point[1]*10**6
            if len(ref_point) == 2:
                self.cats.SetRefPosition(p, 0)
            else:
                self.cats.SetRefPosition(p, ref_point[2])
         
    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be recreated
        
        Probably want to return filename as well  
        """
        repr_ = '{0}(scale_type={1.scale_type}, scale_value={1.scale_value}, diffusion_coefficient={1.eddy_diffusion})'.format(self.__class__.__name__, self)
        return repr_
      
    def __str__(self):
        """Return string representation of this object"""
        
        info  = "{0} object - see attributes for more info\n".format(self.__class__.__name__)
        info += "  scale type = {0.scale_type}\n".format(self)
        info += "  scale value = {0.scale_value}\n".format(self)
        info += "  eddy diffusion coef = {0.eddy_diffusion}\n".format(self)
        
        return info
         
    def set_shio(self, CyShioTime cy_shio):
        self.cats.SetTimeDep(cy_shio.shio)
        self.cats.SetRefPosition(cy_shio.shio.GetStationLocation(), 0)
        self.cats.bTimeFileActive = True
        self.cats.scaleType = 1
        return True
        
    def set_ossm(self, CyOSSMTime ossm):
        self.cats.SetTimeDep(ossm.time_dep)
        self.cats.bTimeFileActive = True   # What is this?
        return True
            
    def text_read(self, fname):
        cdef OSErr err
        cdef bytes path_
        
        fname = os.path.normpath(fname)
        path_ = to_bytes( unicode(fname))
        
        if os.path.exists(path_):
            err = self.cats.TextRead(path_)
            if err != False:
                raise ValueError("CATSMover.ReadTopology(..) returned an error. OSErr: {0}".format(err))
        else:
            raise IOError("No such file: " + path_)
        
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
        
