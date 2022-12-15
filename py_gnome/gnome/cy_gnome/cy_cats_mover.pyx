import os

cimport numpy as cnp
import numpy as np
from libc.string cimport memcpy

from .type_defs cimport *
from .utils cimport _GetHandleSize
from .movers cimport Mover_c
from .current_movers cimport CATSMover_c
from .cy_current_mover cimport CyCurrentMover, dc_mover_to_cmover

from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
from gnome.cy_gnome.cy_shio_time cimport CyShioTime
from gnome.cy_gnome.cy_helpers import filename_as_bytes

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition
for each mover
"""
cdef extern from *:
    CATSMover_c* dynamic_cast_ptr "dynamic_cast<CATSMover_c *>" \
        (Mover_c *) except NULL


cdef class CyCatsMover(CyCurrentMover):

    cdef CATSMover_c *cats

    def __cinit__(self):
        'No plans to subclass CATSMover so no check to see if called by child'
        self.mover = new CATSMover_c()
        self.curr_mover = dc_mover_to_cmover(self.mover)
        self.cats = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        # since this is allocated in this class, free memory here as well
        del self.mover
        self.mover = NULL
        self.curr_mover = NULL
        self.cats = NULL

    def __init__(self, scale_type=0, scale_value=1,
                 uncertain_eddy_diffusion=0, uncertain_eddy_v0=.1,
                 ref_point=None, *args, **kwargs):
        """
        Initialize the CyCatsMover which sets the properties for the underlying
        C++ CATSMover_c object

        :param scale_type=0: There are 3 options in c++, however only
            two options are used:
                - SCALE_NONE = 0
                - SCALE_CONSTANT = 1
            The python CatsMover wrapper sets only 0 or 1. Default is NONE.
        :param scale_value=1: The value by which to scale the data.
            By default, this is 1 which means no scaling
        :param uncertain_eddy_diffusion: Diffusion coefficient for
            eddy diffusion. Default is 0.
        :param uncertain_eddy_v0: Default is .1 (Check that this is still used)
        :param ref_point: Reference point used by C++ CATSMover_c.
            Default (long, lat, z) = (0, 0, -999)

        .. note:: See base class for remaining properties which can be given
        as *args, or **kwargs. The *args is for pickling to work since it
        doesn't understand kwargs.
        """
        self.cats.scaleType = scale_type
        self.cats.scaleValue = scale_value
        self.cats.fEddyDiffusion = uncertain_eddy_diffusion

        if ref_point is not None:
            # defaults (0, 0, -999)
            self.ref_point = ref_point

        super(CyCatsMover, self).__init__(*args, **kwargs)
        # # should not have to do this manually.
        # # make-shifting for now.
        # self.cats.fOptimize.isOptimizedForStep = 0
        # self.cats.fOptimize.isFirstStep = 1

    def __reduce__(self):
        super_reduce = super(CyCatsMover, self).__reduce__()
        return (
            CyCatsMover,
            (
                self.cats.scaleType,
                self.cats.scaleValue,
                self.cats.fEddyDiffusion,
                0.1,
                self.ref_point,
            ) + super_reduce[1]
        )

    property scale_type:
        def __get__(self):
            return self.cats.scaleType

        def __set__(self, value):
            'This should be 0 or 1'
            self.cats.scaleType = value

    property scale_value:
        def __get__(self):
            return self.cats.scaleValue

        def __set__(self, value):
            self.cats.scaleValue = value

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

    property ref_scale:
        def __get__(self):
            return self.cats.refScale

    property ref_point:
        def __get__(self):
            """
            returns the tuple containing (long, lat, z) of reference point
            if it is defined either by the user or obtained from the
            Shio object; otherwise it returns None

            TODO: make sure this is consistent with the format of
                  CyShioTime.ref_point
            """
            ref = self.cats.GetRefPosition()
            if np.isclose(ref.p.pLat, -999.):
                return None
            else:
                return (ref.p.pLong / 1.e6, ref.p.pLat / 1.e6, ref.z)

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
            cdef WorldPoint3D pos

            if not isinstance(ref_point, (list, tuple)) or len(ref_point) != 3:
                raise ValueError('ref_point needs to be '
                                 'in the format (long, lat, z)')

            pos.p.pLong = ref_point[0] * 10 ** 6  # should this happen in C++?
            pos.p.pLat = ref_point[1] * 10 ** 6
            pos.z = ref_point[2]

            self.cats.SetRefPosition(pos)

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated

        Probably want to return filename as well
        """
        b_repr = super(CyCatsMover, self).__repr__()
        b_add = b_repr[b_repr.find('(') + 1:]
        c_repr = ('{0.__class__.__name__}(scale_type={0.scale_type}, '
                  'scale_value={0.scale_value}, '
                  'uncertain_eddy_diffusion={0.uncertain_eddy_diffusion}, '
                  'uncertain_eddy_v0={0.uncertain_eddy_v0}, ').format(self)
        # append ref_point and base class props:
        c_repr += 'ref_point=%s, ' % str(self.ref_point) + b_add

        return c_repr

    def __str__(self):
        """Return string representation of this object"""
        b_str = super(CyCatsMover, self).__str__()
        c_str = b_str + ('  scale type = {0.scale_type}\n'
                         '  scale value = {0.scale_value}\n'
                         '  eddy diffusion coef={0.uncertain_eddy_diffusion}\n'
                         '  ref_point={0.ref_point}\n'
                         .format(self))

        return c_str

    def set_shio(self, CyShioTime cy_shio):
        """
        Takes a CyShioTime object as input and sets C++ Cats mover properties
        from the Shio object.
        """
        self.cats.SetTimeDep(cy_shio.shio)

        if cy_shio.station_location is not None and self.ref_point is None:
            self.ref_point = cy_shio.station_location

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

        if ossm.station_location is not None and self.ref_point is None:
            self.ref_point = ossm.station_location

        return True

    def unset_tide(self):
        """
        Undoes the above tide functions and returns the object to default
        """
        self.cats.SetTimeDep(NULL)
        self.cats.bTimeFileActive = False
        self.cats.scaleType = 0

        self.ref_point = (-0.000999,-0.000999,-0.000999)

    def text_read(self, fname):
        """
        :param fname: path of the file to be read
        :type fname: PathLike

        read the current file
        """
        cdef OSErr err
        cdef bytes  path_ = filename_as_bytes(fname)

        err = self.cats.TextRead(path_)
        if err is not False:
            raise ValueError('CATSMover.text_read(..) '
                             'returned an error. OSErr: {0}'
                             .format(err))

        return True

    def compute_velocity_scale(self):
        """
        compute velocity scale
        """
        cdef OSErr err

        err = self.cats.InitMover()
        if err is not False:
            raise ValueError('CATSMover.compute_velocity_scale(..) '
                             'returned an error. Reference point not valid. OSErr: {0}'
                             .format(err))

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
        :param le_status: Status of each particle - movement is only on
                          particles in water
        :param spill_type: LEType defining whether spill is forecast
                           or uncertain
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

    # ==========================================================================
    # TODO: What are these used for
    # def compute_velocity_scale(self, model_time):
    #    self.mover.ComputeVelocityScale(model_time)
    #
    # def set_velocity_scale(self, scale_value):
    #    self.mover.refScale = scale_value
    # ==========================================================================

    def _get_velocity_handle(self):
        """
            Invokes the GetVelocityHdl method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(VelocityFRec)
        cdef VelocityFH vel_hdl
        cdef cnp.ndarray[VelocityFRec, ndim = 1] vels

        # allocate memory and copy it over
        vel_hdl = self.cats.GetVelocityHdl()
        sz = _GetHandleSize(<Handle>vel_hdl)

        # will this always work?
        vels = np.empty((sz // tmp_size,), dtype=basic_types.velocity_rec)

        memcpy(&vels[0], vel_hdl[0], sz)

        return vels

    def _get_center_points(self):
        """
            Invokes the GetTriangleCenters method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(WorldPoint)
        cdef WORLDPOINTH pts_hdl
        cdef cnp.ndarray[WorldPoint, ndim = 1] pts

        # allocate memory and copy it over
        pts_hdl = self.cats.GetTriangleCenters()
        sz = _GetHandleSize(<Handle>pts_hdl)

        # will this always work?
        pts = np.empty((sz // tmp_size,), dtype=basic_types.w_point_2d)

        memcpy(&pts[0], pts_hdl[0], sz)

        return pts

    def _get_world_points(self):
        """
            Invokes the GetTriangleCenters method of TriGridVel_c object
            to get the velocities for the grid
        """
        cdef short tmp_size = sizeof(WorldPoint)
        cdef WORLDPOINTH pts_hdl
        cdef cnp.ndarray[WorldPoint, ndim = 1] pts

        # allocate memory and copy it over
        pts_hdl = self.cats.GetWorldPointsHdl()
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
        pts_hdl = self.cats.GetPointsHdl()
        sz = _GetHandleSize(<Handle>pts_hdl)

        # will this always work?
        pts = np.empty((sz // tmp_size,), dtype=basic_types.long_point)

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
        top_hdl = self.cats.GetTopologyHdl()
        sz = _GetHandleSize(<Handle>top_hdl)

        # will this always work?
        top = np.empty((sz // tmp_size,), dtype=basic_types.triangle_data)

        memcpy(&top[0], top_hdl[0], sz)

        return top

    def _get_bounds(self):
        """
            Invokes the GetBounds method of TriGridVel_c object
            to get the grid bounds
        """
        bounds = self.cats.GetGridBounds()

        return bounds

