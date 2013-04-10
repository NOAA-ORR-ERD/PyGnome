cimport numpy as cnp
from gnome import basic_types

from type_defs cimport OSErr


cdef class CyMover(object):
    """
    Class serves as a base class for cython wrappers around C++ movers. The C++ movers derive
    from Mover_c.cpp. This provides the default implementation for prepare_for_model_run, prepare_for_model_step
    and model_step_is_done.
    
    In general, the cython wrappers (cy_*) will instantiate the correct C++ object, say cy_wind_mover
    instantiates self.mover as a WindMover_c object. 
    
    It would be desirable to make it so the user cannot instantiate an object of type CyMover, so it only
    serves as a base class; however, since I don't believe I can do that - CyMover.__init__() creates
    a new Mover_c object. This is so the application doesn't crash if the user instantiates a CyMover object
    in Python. Though this object doesn't do much and it does not have a get_move method, it doesn't crash either.
    """
    def __init__(self):
        """
        By default it instantiates Mover_c object. This is only so Python doesn't crash if user
        instantiates a CyMover object in Python. Though the main purpose of this class is to serve
        as a base class to all the cython wrappers around the C++ movers.
        """
        self.mover = new Mover_c()
    
    def prepare_for_model_run(self):
        """
        default implementation. It calls the C++ objects's PrepareForModelRun() method
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
            raise OSError("PrepareForModelStep returned an error: {0}".format(err))
        
    def model_step_is_done(self):
        """
        .. function:: model_step_is_done()
        
        Default call to C++ ModelStepIsDone method
        """
        self.mover.ModelStepIsDone()