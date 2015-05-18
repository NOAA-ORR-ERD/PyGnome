
from libc.stdint cimport int32_t

cimport numpy as cnp

from type_defs cimport OSErr, Seconds

from gnome import basic_types


cdef class CyMover(object):
    """
    Class serves as a base class for cython wrappers around C++ movers.
    The C++ movers derive from Mover_c.cpp. This provides the default
    implementation for prepare_for_model_run, prepare_for_model_step
    and model_step_is_done.

    In general, the cython wrappers (cy_*) will instantiate the correct
    C++ object, say cy_wind_mover instantiates self.mover as a WindMover_c
    object.

    It would be desirable to make it so the user cannot instantiate an object
    of type CyMover, so it only serves as a base class; however, since I don't
    believe I can do that - CyMover.__init__() sets self.mover = NULL
    This is so the application doesn't crash if the user instantiates a
    CyMover object in Python. Though this object doesn't do anything and it
    does not have a get_move method.
    """
    def __cinit__(self):
        '''
        initialize mover to NULL
        '''
        self.mover = NULL

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated
        """
        return '{0}()'.format(self.__class__.__name__)

    def __str__(self):
        """Return string representation of this object"""
        return ('{0} object - see attributes for more info'
                .format(self.__class__.__name__))

    def prepare_for_model_run(self):
        """
        default implementation. It calls the C++ objects's
        PrepareForModelRun() method
        """
        if self.mover:
            self.mover.PrepareForModelRun()

    def prepare_for_model_step(self,
                               Seconds model_time,
                               Seconds step_len,
                               numSets=0,
                               cnp.ndarray[int32_t] setSizes=None):
        """
        .. function:: prepare_for_model_step(self, model_time, step_len,
                                             uncertain)

        prepares the mover for time step, calls the underlying C++ mover
        objects PrepareForModelStep(..)

        :param model_time: current model time.
        :param step_len: length of the time step over which the get move
                         will be computed
        :param numSets: either 0 or 1 if uncertainty is on.
        :param setSizes: Numpy array containing dtype=int for the size of
                         uncertainty array if numSets is 1
        """
        cdef OSErr err
        if self.mover:
            if numSets == 0:
                err = self.mover.PrepareForModelStep(model_time, step_len,
                                                     False, 0, NULL)
            else:
                err = self.mover.PrepareForModelStep(model_time, step_len,
                                                     True, numSets,
                                                     &setSizes[0])

            if err != 0:
                """
                For now just raise an OSError - until the types of possible
                errors are defined and enumerated
                """
                raise OSError("{0.__class__.__name__} returned an error: {1}"
                              .format(self, err))

    def model_step_is_done(self, cnp.ndarray[short] LE_status=None):
        """
        .. function:: model_step_is_done()

        Default call to C++ ModelStepIsDone method
        If model is uncertain remove particles with to_be_removed status from
        uncertainty array

        :para LE_status: numpy array containing the LE_status for each
              released LE.
        """
        cdef OSErr err
        if LE_status is None:
            num_LEs = 0
        else:
            num_LEs = len(LE_status)

        if self.mover:
            if num_LEs > 0:
                err = self.mover.ReallocateUncertainty(num_LEs, &LE_status[0])
        if self.mover:
            self.mover.ModelStepIsDone()


cdef class CyWindMoverBase(CyMover):

    def __cinit__(self):
        '''
        Wind object must be defined by one of the children
        '''
        self.wind = NULL

    def __init__(self, uncertain_duration=10800, uncertain_time_delay=0,
                 uncertain_speed_scale=2, uncertain_angle_scale=0.4):
        """
        .. function:: __init__(self, uncertain_duration=10800, uncertain_time_delay=0,
                 uncertain_speed_scale=2, uncertain_angle_scale=0.4)

        initialize a constant wind mover

        :param uncertain_duation: time in seconds after which the uncertainty
            values are updated
        :param uncertain_time_delay: wait this long after model_start_time to
            turn on uncertainty
        :param uncertain_speed_scale: used in uncertainty computation
        :param uncertain_angle_scale: used in uncertainty computation

        constant_wind_value is a tuple of values: (u, v)
        """

        # Assume wind is constant, but initialize velocity to 0.
        self.wind.fIsConstantWind = 1
        self.wind.fConstantValue.u = 0
        self.wind.fConstantValue.v = 0
        self.wind.fDuration = uncertain_duration
        self.wind.fUncertainStartTime = uncertain_time_delay
        self.wind.fSpeedScale = uncertain_speed_scale
        self.wind.fAngleScale = uncertain_angle_scale

    property uncertain_duration:
        def __get__(self):
            return self.wind.fDuration

        def __set__(self, value):
            self.wind.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.wind.fUncertainStartTime

        def __set__(self, value):
            self.wind.fUncertainStartTime = value

    property uncertain_speed_scale:
        def __get__(self):
            return self.wind.fSpeedScale

        def __set__(self, value):
            self.wind.fSpeedScale = value

    property uncertain_angle_scale:
        def __get__(self):
            return self.wind.fAngleScale

        def __set__(self, value):
            self.wind.fAngleScale = value


cdef class CyCurrentMoverBase(CyMover):
    '''
    Base class - expect children to instantiate/destroy the C++ current mover
    object. This keeps common properties.
    Don't expect python to create/use a C++ CurrentMover_c object
    '''
    def __cinit__(self):
        self.curr_mv = NULL

    def __init__(self,
                 uncertain_duration=172800,
                 uncertain_time_delay=0,
                 up_cur_uncertain=.3,
                 down_cur_uncertain=-.3,
                 right_cur_uncertain=.1,
                 left_cur_uncertain=-.1,
                 ):
        '''
        :param uncertain_duration: how often does a given uncertain element
            get reset. Default (48 hours = 48*3600 sec)
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param up_cur_uncertain: Scale for uncertainty along the flow
        :param down_cur_uncertain: Scale for uncertainty along the flow
        :param right_cur_uncertain: Scale for uncertainty across the flow
        :param left_cur_uncertain: Scale for uncertainty across the flow
        '''
        # move following two to Mover base class
        self.curr_mv.fDuration = uncertain_duration
        self.curr_mv.fUncertainStartTime = uncertain_time_delay

        self.curr_mv.fUpCurUncertainty = up_cur_uncertain
        self.curr_mv.fDownCurUncertainty = down_cur_uncertain
        self.curr_mv.fLeftCurUncertainty = left_cur_uncertain
        self.curr_mv.fRightCurUncertainty = right_cur_uncertain

    property uncertain_duration:
        def __get__(self):
            return self.curr_mv.fDuration

        def __set__(self, value):
            self.curr_mv.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.curr_mv.fUncertainStartTime

        def __set__(self, value):
            self.curr_mv.fUncertainStartTime = value

    property up_cur_uncertain:
        def __get__(self):
            return self.curr_mv.fUpCurUncertainty

        def __set__(self, value):
            self.curr_mv.fUpCurUncertainty = value

    property down_cur_uncertain:
        def __get__(self):
            return self.curr_mv.fDownCurUncertainty

        def __set__(self, value):
            self.curr_mv.fDownCurUncertainty = value

    property right_cur_uncertain:
        def __get__(self):
            return self.curr_mv.fRightCurUncertainty

        def __set__(self, value):
            self.curr_mv.fRightCurUncertainty = value

    property left_cur_uncertain:
        def __get__(self):
            return self.curr_mv.fLeftCurUncertainty

        def __set__(self, value):
            self.curr_mv.fLeftCurUncertainty = value

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated
        """
        return ('{0.__class__.__name__}('
                'uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'up_cur_uncertain={0.up_cur_uncertain}, '
                'down_cur_uncertain={0.down_cur_uncertain}, '
                'right_cur_uncertain={0.right_cur_uncertain}, '
                'left_cur_uncertain={0.left_cur_uncertain})'
                .format(self))

    def __str__(self):
        """Return string representation of this object"""
        return ('{0.__class__.__name__} object '
                '- see attributes for more info\n'
                '  uncertain_duration = {0.uncertain_duration}\n'
                '  uncertain_time_delay = {0.uncertain_time_delay}\n'
                '  up_cur_uncertain = {0.up_cur_uncertain}\n'
                '  down_cur_uncertain = {0.down_cur_uncertain}\n'
                '  right_cur_uncertain = {0.right_cur_uncertain}\n'
                '  left_cur_uncertain = {0.left_cur_uncertain}\n'
                .format(self))