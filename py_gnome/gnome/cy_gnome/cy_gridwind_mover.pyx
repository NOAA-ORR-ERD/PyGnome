cimport numpy as cnp
import numpy as np
import os

from type_defs cimport *
from movers cimport Mover_c,GridWindMover_c,TimeGridVel_c
cimport cy_mover

from gnome.cy_gnome.cy_helpers cimport to_bytes

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition for each mover
"""
cdef extern from *:
    GridWindMover_c* dynamic_cast_ptr "dynamic_cast<GridWindMover_c *>" (Mover_c *) except NULL

cdef class CyGridWindMover(cy_mover.CyMover):

    cdef GridWindMover_c *grid_wind

    def __cinit__(self):
        self.mover = new GridWindMover_c()
        self.grid_wind = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        del self.mover
        self.grid_wind = NULL

    property wind_scale:
        def __get__(self):
            return self.grid_wind.fWindScale
        
        def __set__(self, value):
            self.grid_wind.fWindScale = value
        
    property uncertain_duration:
        def __get__(self):
            return self.grid_wind.fDuration

        def __set__(self, value):
            self.grid_wind.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.grid_wind.fUncertainStartTime

        def __set__(self, value):
            self.grid_wind.fUncertainStartTime = value

    property uncertain_speed_scale:
        def __get__(self):
            return self.grid_wind.fSpeedScale

        def __set__(self, value):
            self.grid_wind.fSpeedScale = value

    property uncertain_angle_scale:
        def __get__(self):
            return self.grid_wind.fAngleScale

        def __set__(self, value):
            self.grid_wind.fAngleScale = value

    def text_read(self, time_grid_file, topology_file=None):
        """
        .. function::text_read

        """
        cdef OSErr err
        cdef bytes time_grid, topology

        time_grid_file = os.path.normpath(time_grid_file)
        time_grid = to_bytes(unicode(time_grid_file))

        if topology_file is None:
            err = self.grid_wind.TextRead(time_grid, '')
        else:
            topology_file = os.path.normpath(topology_file)
            topology = to_bytes(unicode(topology_file))
            err = self.grid_wind.TextRead(time_grid, topology)

        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("GridWindMover_c.TextRead returned an error.")

    def export_topology(self, topology_file):
        """
        .. function::export_topology

        """
        cdef OSErr err
        topology_file = os.path.normpath(topology_file)
        topology_file = to_bytes(unicode(topology_file))
        err = self.grid_wind.ExportTopology(topology_file)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("GridWindMover_c.ExportTopology returned an error.")

    def __init__(self, wind_scale=1):
        """
        .. function:: __init__(self, wind_scale=1)
        
        initialize a grid wind mover
        
        :param wind_scale: scale factor applied to wind values
        
        """
        self.grid_wind.fWindScale = wind_scale
        self.grid_wind.fIsOptimizedForStep = 0

    property extrapolate:
        def __get__(self):
            return self.grid_wind.GetExtrapolationInTime()
        
        def __set__(self, value):
            self.grid_wind.SetExtrapolationInTime(value)
        
    property time_offset:
        def __get__(self):
            return self.grid_wind.GetTimeShift()
        
        def __set__(self, value):
            self.grid_wind.SetTimeShift(value)
        
    def extrapolate_in_time(self, extrapolate):
        self.grid_wind.SetExtrapolationInTime(extrapolate)

    def offset_time(self, time_offset):
        self.grid_wind.SetTimeShift(time_offset)

    def get_move(self,
                 model_time,
                 step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] windages,
                 cnp.ndarray[cnp.npy_int16] LE_status,
                 LEType spill_type):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_double] windages,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type)

        Invokes the underlying C++ GridWindMover_c.get_move(...)

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
        :param spill_type: LEType defining whether spill is forecast or
            uncertain
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points)

        err = self.grid_wind.get_move(N, model_time, step_len, &ref_points[0],
                            &delta[0], &windages[0], <short *>&LE_status[0],
                            spill_type, 0)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points and"
                             " delta are defined")

        """
        Can probably raise this error before calling the C++ code - but the
        C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be"
                             " 'forecast' or 'uncertainty' - you've chosen: "
                             + str(spill_type))
