import os

cimport numpy as cnp
import numpy as np

from libc.string cimport memcpy

from .type_defs cimport *

from gnome import basic_types

from .utils cimport _GetHandleSize

from gnome.cy_gnome.cy_helpers import filename_as_bytes
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime

from .movers cimport Mover_c
from .current_movers cimport CurrentCycleMover_c
from gnome.cy_gnome.cy_mover cimport CyMover


cdef extern from *:
    CurrentCycleMover_c* dynamic_cast_ptr "dynamic_cast<CurrentCycleMover_c *>" (Mover_c *) except NULL


cdef class CyCurrentCycleMover(CyMover):
    # note - current_cycle is derived from grid_current
    def __cinit__(self):
        self.mover = new CurrentCycleMover_c()
        self.current_cycle = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        del self.mover
        self.current_cycle = NULL

    def text_read(self, time_grid_file, topology_file=None):
        """
        .. function::text_read

        """
        cdef OSErr err
        cdef bytes time_grid, topology

        time_grid = filename_as_bytes(time_grid_file)

        if topology_file is None:
            err = self.current_cycle.TextRead(time_grid, '')
        else:
            topology =  filename_as_bytes(topology_file)
            err = self.current_cycle.TextRead(time_grid, topology)

        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("{0}.TextRead returned an error."
                          .format(self.__class__.__name__))

    def __init__(self, current_scale=1,
                 uncertain_duration=24*3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_cross=.25,
                 num_method='Euler'):
        """
        .. function:: __init__(self, current_scale=1,
                               uncertain_duration=24*3600,
                               uncertain_time_delay=0,
                               uncertain_along=.5,
                               uncertain_cross = .25)

        initialize a current cycle mover

        :param uncertain_duation: time in seconds after which the
                                  uncertainty values are updated
        :param uncertain_time_delay: wait this long after model_start_time
                                     to turn on uncertainty
        :param uncertain_cross: used in uncertainty computation, perpendicular
                                to current flow
        :param uncertain_along: used in uncertainty computation, parallel
                                to current flow
        :param current_scale: scale factor applied to current values
        """
        self.num_method = num_method

        self.current_cycle.fCurScale = current_scale
        # self.current_cycle.fUncertainParams.durationInHrs = uncertain_duration
        self.current_cycle.fDuration = uncertain_duration
        # self.current_cycle.fUncertainParams.startTimeInHrs = uncertain_time_delay
        self.current_cycle.fUncertainStartTime = uncertain_time_delay
        # self.current_cycle.fUncertainParams.crossCurUncertainty = uncertain_cross
        # self.current_cycle.fUncertainParams.alongCurUncertainty = uncertain_along
        self.current_cycle.fDownCurUncertainty = -1*uncertain_along
        self.current_cycle.fUpCurUncertainty = uncertain_along
        self.current_cycle.fLeftCurUncertainty = -1*uncertain_cross
        self.current_cycle.fRightCurUncertainty = uncertain_cross

        self.current_cycle.fIsOptimizedForStep = 0

    def __repr__(self):
        """
        unambiguous repr of object, reuse for str() method
        """
        info = ('CyCurrentCycleMover(uncertain_duration={}, '
                'uncertain_time_delay={}, '
                'uncertain_along={}, uncertain_cross={})'
                .format(self.current_cycle.fDuration,
                        self.current_cycle.fUncertainStartTime,
                        self.current_cycle.fUpCurUncertainty,
                        self.current_cycle.fRightCurUncertainty))
        return info

    def __str__(self):
        """Return string representation of this object"""

        info = ('CyCurrentCycleMover object - \n'
                '\tuncertain_duration: %s \n'
                '\tuncertain_time_delay: %s \n'
                '\tuncertain_along: %s\n'
                '\tuncertain_cross: %s'
                .format(self.current_cycle.fDuration,
                        self.current_cycle.fUncertainStartTime,
                        self.current_cycle.fUpCurUncertainty,
                        self.current_cycle.fRightCurUncertainty))

        return info

    property current_scale:
        def __get__(self):
            return self.current_cycle.fCurScale

        def __set__(self, value):
            self.current_cycle.fCurScale = value

    property uncertain_duration:
        def __get__(self):
            return self.current_cycle.fDuration

        def __set__(self, value):
            self.current_cycle.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.current_cycle.fUncertainStartTime

        def __set__(self, value):
            self.current_cycle.fUncertainStartTime = value

    property uncertain_cross:
        def __get__(self):
            return self.current_cycle.fRightCurUncertainty

        def __set__(self, value):
            self.current_cycle.fRightCurUncertainty = value
            self.current_cycle.fLeftCurUncertainty = -1.*value

    property uncertain_along:
        def __get__(self):
            return self.current_cycle.fUpCurUncertainty

        def __set__(self, value):
            self.current_cycle.fUpCurUncertainty = value
            self.current_cycle.fDownCurUncertainty = -1.*value

    property extrapolate:
        def __get__(self):
            return self.current_cycle.GetExtrapolationInTime()

        def __set__(self, value):
            self.current_cycle.SetExtrapolationInTime(value)

    property time_offset:
        def __get__(self):
            return self.current_cycle.GetTimeShift()

        def __set__(self, value):
            self.current_cycle.SetTimeShift(value)

    property num_method:
        def __get__(self):
            return self._num_method

        def __set__(self, value):
            # _num_method is a char* in C++ code
            cdef bytes bvalue = value.encode('ASCII')
            self._num_method = bvalue
            self.current_cycle.num_method = basic_types.numerical_methods[value]

    def extrapolate_in_time(self, extrapolate):
        self.current_cycle.SetExtrapolationInTime(extrapolate)

    def offset_time(self, time_offset):
        self.current_cycle.SetTimeShift(time_offset)

    def get_offset_time(self):
        return self.current_cycle.GetTimeShift()

    def get_start_time(self):
        cdef OSErr err
        cdef Seconds start_time

        err = self.current_cycle.GetDataStartTime(&start_time)

        return start_time

    def get_end_time(self):
        cdef OSErr err
        cdef Seconds end_time

        err = self.current_cycle.GetDataEndTime(&end_time)

        return end_time

    def set_shio(self, CyShioTime cy_shio):
        """
        Takes a CyShioTime object as input and sets C++ CurrentCycle
        mover properties from the Shio object.
        """
        self.current_cycle.SetTimeDep(cy_shio.shio)
        self.current_cycle.SetRefPosition(cy_shio.shio.GetStationLocation())
        self.current_cycle.bTimeFileActive = True
        # self.current_cycle.scaleType = 1
        return True

    def set_ossm(self, CyOSSMTime ossm):
        """
        Takes a CyOSSMTime object as input and sets C++ CurrentCycle
        mover properties from the OSSM object.
        """
        self.current_cycle.SetTimeDep(ossm.time_dep)
        self.current_cycle.bTimeFileActive = True   # What is this?
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
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type)

        Invokes the underlying C++ CurrentCycleMover_c.get_move(...)

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

        err = self.current_cycle.get_move(N, model_time, step_len,
                                          &ref_points[0],
                                          &delta[0],
                                          &LE_status[0],
                                          spill_type,
                                          0)

        if err == 1:
            raise ValueError('Make sure numpy arrays for ref_points '
                             'and delta are defined')

        """
        Can probably raise this error before calling the C++ code,
        but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError('The value for spill type can only be '
                             '"forecast" or "uncertainty" - '
                             'you\'ve chosen: {}'
                             .format(spill_type))

    def _get_triangle_data(self):
        """
            Invokes the GetToplogyHdl method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(Topology)
        cdef TopologyHdl top_hdl
        cdef cnp.ndarray[Topology, ndim = 1] top

        # allocate memory and copy it over
        # should check that topology exists
        top_hdl = self.current_cycle.GetTopologyHdl()
        sz = _GetHandleSize(<Handle>top_hdl)

        # will this always work?
        top = np.empty((sz // tmp_size,), dtype=basic_types.triangle_data)

        memcpy(&top[0], top_hdl[0], sz)

        return top

    def _get_center_points(self):
        """
            Invokes the GetTriangleCenters method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(WorldPoint)
        cdef WORLDPOINTH pts_hdl
        cdef cnp.ndarray[WorldPoint, ndim = 1] pts

        # allocate memory and copy it over
        pts_hdl = self.current_cycle.GetTriangleCenters()
        sz = _GetHandleSize(<Handle>pts_hdl)

        # will this always work?
        pts = np.empty((sz // tmp_size,), dtype=basic_types.w_point_2d)

        memcpy(&pts[0], pts_hdl[0], sz)

        return pts

    def _get_points(self):
        """
            Invokes the GetPointsHdl method of TriGridVel_c object
            to get the points for the grid
        """
        cdef short tmp_size = sizeof(LongPoint)
        cdef LongPointHdl pts_hdl
        cdef cnp.ndarray[LongPoint, ndim = 1] pts

        # allocate memory and copy it over
        pts_hdl = self.current_cycle.GetPointsHdl()
        sz = _GetHandleSize(<Handle>pts_hdl)

        # will this always work?
        pts = np.empty((sz // tmp_size,), dtype=basic_types.long_point)

        memcpy(&pts[0], pts_hdl[0], sz)

        return pts

    def _get_center_points(self):
        """
            Invokes the GetTriangleCenters method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(WorldPoint)
        cdef WORLDPOINTH pts_hdl
        cdef cnp.ndarray[WorldPoint, ndim = 1] pts

        # allocate memory and copy it over
        pts_hdl = self.current_cycle.GetTriangleCenters()
        sz = _GetHandleSize(<Handle>pts_hdl)

        # will this always work?
        pts = np.empty((sz // tmp_size,), dtype=basic_types.w_point_2d)

        memcpy(&pts[0], pts_hdl[0], sz)

        return pts

    def _get_triangle_data(self):
        """
            Invokes the GetToplogyHdl method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(Topology)
        cdef TopologyHdl top_hdl
        cdef cnp.ndarray[Topology, ndim = 1] top

        # allocate memory and copy it over
        # should check that topology exists
        top_hdl = self.current_cycle.GetTopologyHdl()
        sz = _GetHandleSize(<Handle>top_hdl)

        # will this always work?
        top = np.empty((sz // tmp_size,), dtype=basic_types.triangle_data)

        memcpy(&top[0], top_hdl[0], sz)

        return top

    def _get_cell_data(self):
        """
            Invokes the GetCellDataHdl method of TimeGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(GridCellInfo)
        cdef GridCellInfoHdl cell_data_hdl
        cdef cnp.ndarray[GridCellInfo, ndim = 1] cell_data

        # allocate memory and copy it over
        # should check that cell data exists
        cell_data_hdl = self.current_cycle.GetCellDataHdl()
        sz = _GetHandleSize(<Handle>cell_data_hdl)

        # will this always work?
        cell_data = np.empty((sz // tmp_size,), dtype=basic_types.cell_data)

        memcpy(&cell_data[0], cell_data_hdl[0], sz)

        return cell_data

    def _is_triangle_grid(self):
        """
            Invokes the IsTriangleGrid TimeGridVel_c object
        """
        return self.current_cycle.IsTriangleGrid()

    def _is_data_on_cells(self):
        """
            Invokes the IsDataOnCells TimeGridVel_c object
        """
        return self.current_cycle.IsDataOnCells()

    def get_num_triangles(self):
        """
            Invokes the GetNumTriangles method of TriGridVel_c object
            to get the number of triangles
        """
        num_tri = self.current_cycle.GetNumTriangles()

        return num_tri

    def _is_regular_grid(self):
        """
            Invokes the IsRegularGrid TimeGridVel_c object
        """
        return self.current_cycle.IsRegularGrid()

    def get_num_points(self):
        """
            Invokes the GetNumPoints method of TriGridVel_c object
            to get the number of triangles
        """
        num_pts = self.current_cycle.GetNumPoints()

        return num_pts

    def _get_bounds(self):
        """
            Invokes the GetBounds method of TriGridVel_c object
            to get the grid bounds
        """
        bounds = self.current_cycle.GetGridBounds()

        return bounds

    def get_scaled_velocities(self, Seconds model_time,
                              cnp.ndarray[VelocityFRec] vels):
        """
            Invokes the GetScaledVelocities method of TimeGridVel_c object
            to get the velocities on the triangles
        """
        cdef OSErr err

        err = self.current_cycle.GetScaledVelocities(model_time, &vels[0])
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError('CurrentCycleMover_c.GetScaledVelocities '
                          'returned an error.')
