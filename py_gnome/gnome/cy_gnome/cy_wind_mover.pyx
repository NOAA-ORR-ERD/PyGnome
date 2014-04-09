cimport numpy as cnp
import numpy as np

from gnome import basic_types

# following exist in gnome.cy_gnome
from movers cimport WindMover_c, Mover_c
from type_defs cimport WorldPoint3D, LEWindUncertainRec, LEStatus, LEType, \
                       OSErr, Seconds, VelocityRec
cimport cy_mover, cy_ossm_time

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition
for each mover
"""
cdef extern from *:
    WindMover_c * dynamic_cast_ptr "dynamic_cast<WindMover_c *>" (Mover_c *) except NULL


cdef class CyWindMover(cy_mover.CyMover):
    """
    Cython wrapper for C++ WindMover_c object.
    It derives from cy_mover.CyMover base class which defines default
    functionality for some methods
    """
    cdef WindMover_c * wind

    def __cinit__(self):
        """
        Create a new WindMover_c() and also do a dynamic cast of
        member variable 'self.mover' to WindMover_c
        so all members of WindMover_c are available to this class
        """
        self.mover = new WindMover_c()
        self.wind = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        # since this is allocated in this class, free memory here as well
        del self.mover
        self.wind = NULL

    def __init__(self, uncertain_duration=10800, uncertain_time_delay=0,
                 uncertain_speed_scale=2, uncertain_angle_scale=0.4):
        """
        .. function:: __init__(self, uncertain_duration=10800,
                               uncertain_time_delay=0,
                               uncertain_speed_scale=2,
                               uncertain_angle_scale=0.4)

        initialize a constant wind mover

        :param uncertain_duation: time in seconds after which the
                                  uncertainty values are updated
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

    def __repr__(self):
        '''
        Return an unambiguous representation of this object so it can be
        recreated via eval()
        '''
        info = ('{0.__class__.__module__}.{0.__class__.__name__}('
                'uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay},'
                'uncertain_speed_scale={0.uncertain_speed_scale}, '
                'uncertain_angle_scale={0.uncertain_angle_scale}'
                ')'.format(self))
        return info

    def __str__(self):
        'Return string representation of this object'
        info = ("{0.__class__.__name__} object - \n"
                "  uncertain_duration: {0.uncertain_duration} \n"
                "  uncertain_time_delay: {0.uncertain_time_delay} \n"
                "  uncertain_speed_scale: {0.uncertain_speed_scale}\n"
                "  uncertain_angle_scale: {0.uncertain_angle_scale}".format(self))

        return info

    def __reduce__(self):
        return (CyWindMover, (self.uncertain_duration,
                              self.uncertain_time_delay,
                              self.uncertain_speed_scale,
                              self.uncertain_angle_scale))

    def __eq(self, CyWindMover other):
        scalar_attrs = ('uncertain_duration', 'uncertain_time_delay',
                        'uncertain_speed_scale', 'uncertain_angle_scale')
        for a in scalar_attrs:
            if not getattr(self, a) == getattr(other, a):
                return False

        return True

    def __richcmp__(self, CyWindMover other, int cmp):
        if cmp not in (2, 3):
            raise NotImplemented('CyWindMover does not support '
                                 'this type of comparison.')

        if cmp == 2:
            return self.__eq(other)
        elif cmp == 3:
            return not self.__eq(other)

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

    def get_move(self, model_time, step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] windages,
                 # TODO: would be nice if we could define this as LEStatus type
                 cnp.ndarray[short] LE_status,
                 LEType spill_type):
        """
        .. function:: get_move(self, model_time, step_len,
                               cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                               cnp.ndarray[WorldPoint3D, ndim=1] delta,
                               cnp.ndarray[cnp.npy_double] windages,
                               cnp.ndarray[cnp.npy_int16] LE_status,
                               LE_type)

        Invokes the underlying C++ WindMover_c.get_move(...)

        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param LE_windage: windage to be applied to each particle
        :type LE_windage: numpy array of numpy.npy_int16
        :param le_status: status of each particle - movement is only on
                          particles in water
        :param spill_type: LEType defining whether spill is forecast
                           or uncertain
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points)  # set a data type?

        # modifies delta in place
        err = self.wind.get_move(N, model_time, step_len,
                                  &ref_points[0],
                                  &delta[0],
                                  &windages[0],
                                  &LE_status[0],
                                  spill_type,
                                  0)
        if err == 1:
            raise ValueError('Make sure numpy arrays for ref_points, delta '
                             'and windages are defined')

        """
        Can probably raise this error before calling the C++ code
        - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' "
                             "or 'uncertainty' - you've chosen: "
                             "{0}".format(spill_type))

    def set_constant_wind(self, windU, windV):
        """
        Constant wind can be set using set_ossm as well; though this is
        exposed for testing
        """
        self.wind.fConstantValue.u = windU
        self.wind.fConstantValue.v = windV
        self.wind.fIsConstantWind = 1

    def set_ossm(self, cy_ossm_time.CyOSSMTime ossm):
        """
        Use the CyOSSMTime object to set the wind mover OSSM time
        member variable using the SetTimeDep method
        """
        self.wind.SetTimeDep(ossm.time_dep)
        self.wind.fIsConstantWind = 0
        return True

    def get_time_value(self, modelTime):
        """
        GetTimeValue - for a specified modelTime or array of model times,
                       it returns the wind velocity values
        """
        cdef cnp.ndarray[Seconds, ndim = 1] modelTimeArray
        modelTimeArray = np.asarray(modelTime,
                                    basic_types.seconds).reshape((-1,))

        # velocity record passed to the methods and returned back to python
        cdef cnp.ndarray[VelocityRec, ndim = 1] vel_rec
        cdef VelocityRec * velrec

        cdef unsigned int i
        cdef OSErr err

        vel_rec = np.empty((modelTimeArray.size,),
                           dtype=basic_types.velocity_rec)

        for i in range(0, modelTimeArray.size):
            err = self.wind.GetTimeValue(modelTimeArray[i], &vel_rec[i])
            if err != 0:
                raise ValueError

        return vel_rec
