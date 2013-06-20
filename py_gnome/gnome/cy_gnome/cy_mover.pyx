cimport numpy as cnp
from gnome import basic_types

from type_defs cimport OSErr, Seconds

cdef class CyMover(object):
    """
    Class serves as a base class for cython wrappers around C++ movers. The C++ movers derive
    from Mover_c.cpp. This provides the default implementation for prepare_for_model_run, prepare_for_model_step
    and model_step_is_done.
    
    In general, the cython wrappers (cy_*) will instantiate the correct C++ object, say cy_wind_mover
    instantiates self.mover as a WindMover_c object. 
    
    It would be desirable to make it so the user cannot instantiate an object of type CyMover, so it only
    serves as a base class; however, since I don't believe I can do that - CyMover.__init__() sets self.mover = NULL 
    This is so the application doesn't crash if the user instantiates a CyMover object
    in Python. Though this object doesn't do anything and it does not have a get_move method.
    """
    def __init__(self):
        """
        By default it sets self.mover=NULL. This is only so Python doesn't crash if user
        instantiates a CyMover object in Python. Though the main purpose of this class is to serve
        as a base class to all the cython wrappers around the C++ movers.
        """
        self.mover = NULL
    
    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be recreated 
        """
        repr_ = '{0}()'.format(self.__class__.__name__)
        return repr_
      
    def __str__(self):
        """Return string representation of this object"""
        
        info  = "{0} object - see attributes for more info".format(self.__class__.__name__)
        
        return info
    
    def prepare_for_model_run(self):
        """
        default implementation. It calls the C++ objects's PrepareForModelRun() method
        """
        if self.mover:
            self.mover.PrepareForModelRun()
    
    def prepare_for_model_step(self, Seconds model_time, Seconds step_len, numSets=0, cnp.ndarray[cnp.npy_long] setSizes=None):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len, uncertain)
        
        prepares the mover for time step, calls the underlying C++ mover objects PrepareForModelStep(..)
        
        :param model_time: current model time.
        :param step_len: length of the time step over which the get move will be computed
        """
        cdef OSErr err
        if self.mover:
            if numSets == 0:
                err = self.mover.PrepareForModelStep(model_time, step_len, False, 0, NULL)
            else:
                err = self.mover.PrepareForModelStep(model_time, step_len, True, numSets, <int *>&setSizes[0])
                
            if err != 0:
                """
                For now just raise an OSError - until the types of possible errors are defined and enumerated
                """
                raise OSError("PrepareForModelStep returned an error: {0}".format(err))
        
    def model_step_is_done(self, cnp.ndarray[cnp.npy_int16] LE_status=None):
        """
        .. function:: model_step_is_done()
        
        Default call to C++ ModelStepIsDone method
        If model is uncertain remove particles with to_be_removed status from uncertainty array
        """
        cdef OSErr err
        if LE_status is None:
            num_LEs = 0
        else:
            num_LEs = len(LE_status)
 
        if self.mover:
            if num_LEs > 0:
                err = self.mover.ReallocateUncertainty(num_LEs, <short *>&LE_status[0])
        if self.mover:
            self.mover.ModelStepIsDone()
