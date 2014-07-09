import os

cimport numpy as cnp
import numpy as np

from type_defs cimport *
from movers cimport CATSMover_c, Mover_c
from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime
from gnome.cy_gnome cimport cy_mover
from gnome.cy_gnome.cy_helpers cimport to_bytes

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition
for each mover
"""
cdef extern from *:
    CATSMover_c* dynamic_cast_ptr "dynamic_cast<CATSMover_c *>" (Mover_c *) except NULL


cdef class CyCatsMover(cy_mover.CyMover):

    cdef CATSMover_c *cats

    def __cinit__(self):
        self.mover = new CATSMover_c()
        self.cats = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        # since this is allocated in this class, free memory here as well
        del self.mover
        self.cats = NULL

    def __init__(self, scale_type=0, scale_value=1,
                 uncertain_duration=48 * 3600, uncertain_time_delay=0,
                 up_cur_uncertain=.3, down_cur_uncertain=-.3,
                 right_cur_uncertain=.1, left_cur_uncertain=-.1,
                 uncertain_eddy_diffusion=0, uncertain_eddy_v0=.1):
        """
        Initialize the CyCatsMover which sets the properties for the underlying C++ CATSMover_c object

        :param scale_type=0: There are 3 options in c++, however only two options are used SCALE_NONE = 0, SCALE_CONSTANT = 1.
                           The python CatsMover wrapper only sets either 0 or 1. Default is NONE.
        :param scale_value=1: The value by which to scale the data. By default, this is 1 which means no scaling
        :param uncertain_duration: how often does a given uncertain element gets reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param up_cur_uncertain: Scale for uncertainty along the flow
        :param down_cur_uncertain: Scale for uncertainty along the flow
        :param right_cur_uncertain: Scale for uncertainty across the flow
        :param left_cur_uncertain: Scale for uncertainty across the flow
        :param uncertain_eddy_diffusion: Diffusion coefficient for eddy diffusion. Default is 0.
        :param uncertain_eddy_v0: Default is .1 (Check that this is still used)
        """
        cdef WorldPoint p
        self.cats.scaleType = scale_type
        self.cats.scaleValue = scale_value
        self.cats.fEddyDiffusion = uncertain_eddy_diffusion
        self.cats.refZ = -999  # default to -1
        ## should not have to do this manually.
        ## make-shifting for now.
        #self.cats.fOptimize.isOptimizedForStep = 0
        #self.cats.fOptimize.isFirstStep = 1

    property scale_type:
        def __get__(self):
            return self.cats.scaleType

        def __set__(self, value):
            self.cats.scaleType = value

    property scale_value:
        def __get__(self):
            return self.cats.scaleValue

        def __set__(self, value):
            self.cats.scaleValue = value

    property uncertain_duration:
        def __get__(self):
            return self.cats.fDuration

        def __set__(self, value):
            self.cats.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.cats.fUncertainStartTime

        def __set__(self, value):
            self.cats.fUncertainStartTime = value

    property up_cur_uncertain:
        def __get__(self):
            return self.cats.fUpCurUncertainty

        def __set__(self, value):
            self.cats.fUpCurUncertainty = value

    property down_cur_uncertain:
        def __get__(self):
            return self.cats.fDownCurUncertainty

        def __set__(self, value):
            self.cats.fDownCurUncertainty = value

    property right_cur_uncertain:
        def __get__(self):
            return self.cats.fRightCurUncertainty

        def __set__(self, value):
            self.cats.fRightCurUncertainty = value

    property left_cur_uncertain:
        def __get__(self):
            return self.cats.fLeftCurUncertainty

        def __set__(self, value):
            self.cats.fLeftCurUncertainty = value

    property uncertain_eddy_diffusion:
        def __get__(self):
            return self.cats.fEddyDiffusion

        def __set__(self, value):
            self.cats.fEddyDiffusion = value

    property uncertain_eddy_v0:
        def __get__(self):
            return self.cats.fEddyV0

        def __set__(self, value):
            self.cats.fEddyV0 = value

    property ref_point:
        def __get__(self):
            """
            returns the tuple containing (long, lat, z) of reference point
            if it is defined either by the user or obtained from the
            Shio object; otherwise it returns None

            TODO: make sure this is consistent with the format of
                  CyShioTime.ref_point
            """
            if self.cats.refZ == -999:
                return None
            else:
                return (self.cats.refP.pLong / 1.e6,
                        self.cats.refP.pLat / 1.e6,
                        self.cats.refZ)

        def __set__(self, ref_point):
            """
            accepts a list or a tuple
            will not work with a numpy array since indexing assumes a list
            or a tuple

            takes only (long, lat, z), if length is bigger than 3, it uses the
            first 3 data points

            TODO: make sure this is consistent with the format of
                  CyShioTime.ref_point
            """
            ref_point = np.asarray(ref_point)  # make it a numpy array
            cdef WorldPoint p
            p.pLong = ref_point[0] * 10 ** 6  # should this happen in C++?
            p.pLat = ref_point[1] * 10 ** 6
            if len(ref_point) == 2:
                self.cats.SetRefPosition(p, 0)
            else:
                self.cats.SetRefPosition(p, ref_point[2])

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated

        Probably want to return filename as well
        """
        return ('{0.__class__.__name__}(scale_type={0.scale_type}, '
                'scale_value={0.scale_value}, '
                'uncertain_eddy_diffusion={0.uncertain_eddy_diffusion})'
                .format(self))

    def __str__(self):
        """Return string representation of this object"""
        return ('{0.__class__.__name__} object '
                '- see attributes for more info\n'
                '  scale type = {0.scale_type}\n'
                '  scale value = {0.scale_value}\n'
                '  eddy diffusion coef = {0.uncertain_eddy_diffusion}\n'
                .format(self))

    def __reduce__(self):
        return (CyCatsMover, (self.scale_type, self.scale_value,
                              self.uncertain_duration,
                              self.uncertain_time_delay,
                              self.up_cur_uncertain,
                              self.down_cur_uncertain,
                              self.right_cur_uncertain,
                              self.left_cur_uncertain,
                              self.uncertain_eddy_diffusion,
                              self.uncertain_eddy_v0))

    def set_shio(self, CyShioTime cy_shio):
        """
        Takes a CyShioTime object as input and sets C++ Cats mover properties
        from the Shio object.
        """
        self.cats.SetTimeDep(cy_shio.shio)
        self.cats.SetRefPosition(cy_shio.shio.GetStationLocation(), 0)
        self.cats.bTimeFileActive = True
        self.cats.scaleType = 1
        return True

    def set_ossm(self, CyOSSMTime ossm):
        """
        Takes a CyOSSMTime object as input and sets C++ Cats mover properties
        from the OSSM object.
        """
        self.cats.SetTimeDep(ossm.time_dep)
        self.cats.bTimeFileActive = True   # What is this?
        return True

    def text_read(self, fname):
        """
        read the current file
        """
        cdef OSErr err
        cdef bytes path_

        fname = os.path.normpath(fname)
        path_ = to_bytes(unicode(fname))

        if os.path.exists(path_):
            err = self.cats.TextRead(path_)
            if err != False:
                raise ValueError('CATSMover.text_read(..) returned an error. '
                                 'OSErr: {0}'.format(err))
        else:
            raise IOError("No such file: " + path_)

        return True

    def get_move(self,
                 model_time,
                 step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[short] LE_status,
                 LEType spill_type):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] windages,
                 cnp.ndarray[short] LE_status,
                 LEType LE_type)

        Invokes the underlying C++ WindMover_c.get_move(...)

        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param LE_windage: windage to be applied to each particle
        :type LE_windage: numpy array of numpy.npy_int16
        :param le_status: status of each particle - movement is only on particles in water
        :param spill_type: LEType defining whether spill is forecast or uncertain 
        :returns: none
        """
        cdef OSErr err

        N = len(ref_points)

        err = self.cats.get_move(N, model_time, step_len,
                                 &ref_points[0], &delta[0], &LE_status[0],
                                 spill_type, 0)
        if err == 1:
            raise ValueError('Make sure numpy arrays for ref_points, delta, '
                             'and windages are defined')

        """
        Can probably raise this error before calling the C++ code
        - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError('The value for spill type can only be '
                             '"forecast" or "uncertainty" '
                             '- you have chosen: {0!s}'.format(spill_type))

    #==========================================================================
    # TODO: What are these used for
    # def compute_velocity_scale(self, model_time):
    #    self.mover.ComputeVelocityScale(model_time)
    #
    # def set_velocity_scale(self, scale_value):
    #    self.mover.refScale = scale_value
    #==========================================================================
