import os

cimport numpy as cnp
import numpy as np
from libc.string cimport memcpy

from .type_defs cimport *
from .utils cimport _GetHandleSize
from .movers cimport Mover_c
from .current_movers cimport IceMover_c, GridCurrentMover_c, CurrentMover_c

from gnome import basic_types
from gnome.cy_gnome.cy_gridcurrent_mover cimport CyGridCurrentMover
from gnome.cy_gnome.cy_helpers import filename_as_bytes


cdef extern from *:
    #IceMover_c* dynamic_cast_ptr "dynamic_cast<IceMover_c *>" (Mover_c *) except NULL
    IceMover_c* dc_mover_to_im "dynamic_cast<IceMover_c *>" (Mover_c *) except NULL
    CurrentMover_c* dc_im_mover_to_cm "dynamic_cast<IceMover_c *>" (Mover_c *) except NULL
    GridCurrentMover_c* dc_im_mover_to_gcm "dynamic_cast<IceMover_c *>" (Mover_c *) except NULL


cdef class CyIceMover(CyGridCurrentMover):

    cdef IceMover_c *grid_ice

    def __cinit__(self):
        self.mover = new IceMover_c()
        #self.grid_ice = dynamic_cast_ptr(self.mover)
        self.grid_ice = dc_mover_to_im(self.mover)
        self.curr_mv = dc_im_mover_to_cm(self.mover)
        self.grid_current = dc_im_mover_to_gcm(self.mover)

    def __dealloc__(self):
        del self.mover
        self.mover = NULL
        self.grid_ice = NULL
        self.curr_mv = NULL
        self.grid_current = NULL

    def text_read(self, time_grid_file, topology_file=None):
        """
        .. function::text_read
        """
        cdef OSErr err
        cdef bytes time_grid, topology

        time_grid = filename_as_bytes(time_grid_file)

        if topology_file is None:
            err = self.grid_ice.TextRead(time_grid, '')
        else:
            topology = filename_as_bytes(topology_file)
            err = self.grid_ice.TextRead(time_grid, topology)

        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("IceMover_c.TextRead returned an error.")

#     def _get_points(self):
#         """
#             Invokes the GetPointsHdl method of TriGridVel_c object
#             to get the points for the grid
#         """
#         cdef short tmp_size = sizeof(LongPoint)
#         cdef LongPointHdl pts_hdl
#         cdef cnp.ndarray[LongPoint, ndim = 1] pts
#
#         # allocate memory and copy it over
#         pts_hdl = self.grid_ice.GetPointsHdl()
#         sz = _GetHandleSize(<Handle>pts_hdl)
#
#         # will this always work?
#         pts = np.empty((sz // tmp_size,), dtype=basic_types.long_point)
#
#         memcpy(&pts[0], pts_hdl[0], sz)
#
#         return pts
#
#     def _get_center_points(self):
#         """
#             Invokes the GetTriangleCenters method of TriGridVel_c object
#             to get the velocities for the grid
#         """
#         cdef short tmp_size = sizeof(WorldPoint)
#         cdef WORLDPOINTH pts_hdl
#         cdef cnp.ndarray[WorldPoint, ndim = 1] pts
#
#         # allocate memory and copy it over
#         pts_hdl = self.grid_ice.GetTriangleCenters()
#         sz = _GetHandleSize(<Handle>pts_hdl)
#
#         # will this always work?
#         pts = np.empty((sz // tmp_size,), dtype=basic_types.w_point_2d)
#
#         memcpy(&pts[0], pts_hdl[0], sz)
#
#         return pts
#
#     def _get_triangle_data(self):
#         """
#             Invokes the GetToplogyHdl method of TriGridVel_c object
#             to get the velocities for the grid
#         """
#         cdef short tmp_size = sizeof(Topology)
#         cdef TopologyHdl top_hdl
#         cdef cnp.ndarray[Topology, ndim = 1] top
#
#         # allocate memory and copy it over
#         # should check that topology exists
#         top_hdl = self.grid_ice.GetTopologyHdl()
#         sz = _GetHandleSize(<Handle>top_hdl)
#
#         # will this always work?
#         top = np.empty((sz // tmp_size,), dtype=basic_types.triangle_data)
#
#         memcpy(&top[0], top_hdl[0], sz)
#
#         return top
#
#     def get_num_triangles(self):
#         """
#             Invokes the GetNumTriangles method of TriGridVel_c object
#             to get the number of triangles
#         """
#         num_tri = self.grid_ice.GetNumTriangles()
#
#         return num_tri
#

    def get_ice_fields(self, Seconds model_time,
                       cnp.ndarray[cnp.npy_double] fraction,
                       cnp.ndarray[cnp.npy_double] thickness):
        """
            Invokes the GetIceFields method of TimeGridVelIce_c object
            to get the fields on the triangles
        """
        cdef OSErr err

        err = self.grid_ice.GetIceFields(model_time,
                                         &fraction[0], &thickness[0])
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("IceMover_c.GetIceFields returned an error.")

    def get_ice_velocities(self, Seconds model_time,
                           cnp.ndarray[VelocityFRec] vels):
        """
            Invokes the GetIceVelocities method of TimeGridVelIce_c object
            to get the velocities on the triangles
        """
        cdef OSErr err

        err = self.grid_ice.GetIceVelocities(model_time, &vels[0])
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("IceMover_c.GetIceVelocities returned an error.")

    def get_movement_velocities(self, Seconds model_time,
                                cnp.ndarray[VelocityFRec] vels):
        """
            Invokes the GetMovementVelocities method of TimeGridVelIce_c object
            to get the velocities on the triangles
        """
        cdef OSErr err

        err = self.grid_ice.GetMovementVelocities(model_time, &vels[0])
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError('IceMover_c.GetMovementVelocities '
                          'returned an error.')
