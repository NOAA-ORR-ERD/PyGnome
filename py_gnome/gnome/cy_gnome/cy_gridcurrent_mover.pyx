cimport numpy as cnp
import numpy as np
import os

from type_defs cimport *
from movers cimport Mover_c
from current_movers cimport GridCurrentMover_c, CurrentMover_c
from gnome.cy_gnome.cy_mover cimport CyCurrentMoverBase
from gnome.cy_gnome.cy_helpers cimport to_bytes


cdef extern from *:
    GridCurrentMover_c* dc_mover_to_gc "dynamic_cast<GridCurrentMover_c *>" \
        (Mover_c *) except NULL
    CurrentMover_c* dc_mover_to_cm "dynamic_cast<GridCurrentMover_c *>" \
        (Mover_c *) except NULL


cdef class CyGridCurrentMover(CyCurrentMoverBase):

    #cdef GridCurrentMover_c *grid_current

    def __cinit__(self):
        self.mover = new GridCurrentMover_c()
        self.grid_current = dc_mover_to_gc(self.mover)
        self.curr_mv = dc_mover_to_cm(self.mover)

    def __dealloc__(self):
        del self.mover
        self.grid_current = NULL
        self.curr_mv = NULL

    def text_read(self, time_grid_file, topology_file=None):
        """
        .. function::text_read

        """
        cdef OSErr err
        cdef bytes time_grid, topology

        time_grid_file = os.path.normpath(time_grid_file)
        time_grid = to_bytes(unicode(time_grid_file))

        if topology_file is None:
            err = self.grid_current.TextRead(time_grid, '')
        else:
            topology_file = os.path.normpath(topology_file)
            topology = to_bytes(unicode(topology_file))
            err = self.grid_current.TextRead(time_grid, topology)

        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("GridCurrentMover_c.TextRead returned an error.")

    def export_topology(self, topology_file):
        """
        .. function::export_topology

        """
        cdef OSErr err
        topology_file = os.path.normpath(topology_file)
        topology_file = to_bytes(unicode(topology_file))
        err = self.grid_current.ExportTopology(topology_file)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("GridCurrentMover_c.ExportTopology returned an error.")

    def __init__(self, current_scale=1,
                 uncertain_duration=24*3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_cross=.25):
        """
        .. function:: __init__(self, current_scale=1, uncertain_duration=24*3600, uncertain_time_delay=0,
                 uncertain_along = .5, uncertain_cross = .25)

        initialize a grid current mover

        :param uncertain_duation: time in seconds after which the uncertainty values are updated
        :param uncertain_time_delay: wait this long after model_start_time to turn on uncertainty
        :param uncertain_cross: used in uncertainty computation, perpendicular to current flow
        :param uncertain_along: used in uncertainty computation, parallel to current flow
        :param current_scale: scale factor applied to current values

        """
        super(CyGridCurrentMover, self).__init__(uncertain_duration=uncertain_duration,
                                                 uncertain_time_delay=uncertain_time_delay,
                                                 up_cur_uncertain=uncertain_along,
                                                 down_cur_uncertain=-1*uncertain_along,
                                                 right_cur_uncertain=uncertain_cross,
                                                 left_cur_uncertain=-1*uncertain_cross)

        self.grid_current.fCurScale = current_scale
        self.grid_current.fIsOptimizedForStep = 0

    def __repr__(self):
        """
        unambiguous repr of object, reuse for str() method
        """
        info = "CyGridCurrentMover(uncertain_duration=%s,uncertain_time_delay=%s,uncertain_along=%s,uncertain_cross=%s)" \
        % (self.grid_current.fDuration, self.grid_current.fUncertainStartTime, self.grid_current.fUpCurUncertainty, self.grid_current.fRightCurUncertainty)
        return info

    def __str__(self):
        """Return string representation of this object"""
        info  = "CyGridCurrentMover object - \n  uncertain_duration: %s \n  uncertain_time_delay: %s \n  uncertain_along: %s\n  uncertain_cross: %s" \
        % (self.grid_current.fDuration, self.grid_current.fUncertainStartTime, self.grid_current.fUpCurUncertainty, self.grid_current.fRightCurUncertainty)
        return info

    property current_scale:
        def __get__(self):
            return self.grid_current.fCurScale

        def __set__(self, value):
            self.grid_current.fCurScale = value

    property uncertain_cross:
        def __get__(self):
            return self.grid_current.fRightCurUncertainty

        def __set__(self, value):
            self.grid_current.fRightCurUncertainty = value
            self.grid_current.fLeftCurUncertainty = -1.*value

    property uncertain_along:
        def __get__(self):
            return self.grid_current.fUpCurUncertainty

        def __set__(self, value):
            self.grid_current.fUpCurUncertainty = value
            self.grid_current.fDownCurUncertainty = -1.*value

    property extrapolate:
        def __get__(self):
            return self.grid_current.GetExtrapolationInTime()

        def __set__(self, value):
            self.grid_current.SetExtrapolationInTime(value)

    property time_offset:
        def __get__(self):
            return self.grid_current.GetTimeShift()

        def __set__(self, value):
            self.grid_current.SetTimeShift(value)

    def extrapolate_in_time(self, extrapolate):
        self.grid_current.SetExtrapolationInTime(extrapolate)

    def offset_time(self, time_offset):
        self.grid_current.SetTimeShift(time_offset)

    def get_offset_time(self):
        return self.grid_current.GetTimeShift()

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
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type)

        Invokes the underlying C++ GridCurrentMover_c.get_move(...)

        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param le_status: status of each particle - movement is only on
                                                    particles in water
        :param spill_type: LEType defining whether spill is forecast
                           or uncertain
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points)

        err = self.grid_current.get_move(N, model_time, step_len,
                                 &ref_points[0],
                                 &delta[0],
                                 &LE_status[0],
                                 spill_type,
                                 0)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points and delta are defined")

        """
        Can probably raise this error before calling the C++ code,
        but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
