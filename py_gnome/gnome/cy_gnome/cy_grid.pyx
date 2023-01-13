from .type_defs cimport OSErr, VelocityRec, WorldPoint3D
from libcpp cimport bool

import cython
cimport numpy as cnp
import numpy as np

from .grids cimport TimeGridVel_c
from gnome.basic_types import velocity_rec

from .cy_helpers import filename_as_bytes

@cython.final
cdef class CyTimeGridVel(object):
    def __init__(self):
        self.timegrid = NULL

    property extrapolate:
        def __get__(self):
            return self.timegrid.fAllowExtrapolationInTime

        def __set__(self, value):
            self.timegrid.fAllowExtrapolationInTime = value

    property time_offset:
        def __get__(self):
            return self.timegrid.fTimeShift

        def __set__(self, value):
            self.timegrid.fTimeShift = value

    def load_data(self, datafile, topology=None):
        """
        load the data from files

        :param datafile: PathLike object to load the data from

        :param topology=None: PathLike object to load topology data.
        """
        cdef OSErr err
        cdef bytes bdatafile
        cdef bytes btopology

        bdatafile = filename_as_bytes(datafile)

        if topology:
            btopology = filename_as_bytes(topology)
            err = self.timegrid.TextRead(bdatafile, btopology)
        else:
            err = self.timegrid.TextRead(bdatafile, '')

        if err != 0:
            raise Exception('Failed in TextRead')

    def set_interval(self, int time):
        cdef OSErr err0
        cdef char errmsg[256]

        err0 = self.timegrid.SetInterval(errmsg, time)
        if err0 != 0:
            raise Exception('SetInterval error message: {0}'.format(errmsg))

    def get_value(self, int time, location):
        cdef WorldPoint3D ref_point
        cdef VelocityRec vel
        cdef OSErr err0
        cdef char errmsg[256]

        location = np.asarray(location)

        # do the multiply by 1000000 here - this is what gnome expects
        ref_point.p.pLong = location[0] * 1000000
        ref_point.p.pLat = location[1] * 1000000

        if len(location) == 2:
            ref_point.z = 0
        else:
            ref_point.z = location[2]

        err0 = self.timegrid.SetInterval(errmsg, time)
        if err0 != 0:
            raise Exception('SetInterval error message: {0}'.format(errmsg))

        vel = self.timegrid.GetScaledPatValue(time, ref_point)

        # return as velocity_rec dtype array
        return np.asarray(tuple(vel.values()), dtype=velocity_rec)

    def get_values(self,
                   int model_time,
                   cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                   cnp.ndarray[VelocityRec] vels):
        """
        .. function:: get_move(self,
                 model_time,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[VelocityRec] vels)

        Invokes the underlying C++ TimeGridVel_c.get_values(...)

        :param model_time: current model time
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param vels: the velocity at the position of each particle
        :type vels: numpy array of VelocityRec
        :returns: none
        """
        cdef OSErr err

        N = len(ref_points)

        err = self.timegrid.get_values(N, model_time,
                                       &ref_points[0], &vels[0])

        if err == 1:
            raise ValueError('Make sure numpy arrays for ref_points and vels, '
                             'are defined')
