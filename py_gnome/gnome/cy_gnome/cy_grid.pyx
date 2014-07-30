from type_defs cimport OSErr, VelocityRec, WorldPoint3D
from libcpp cimport bool

import cython
import numpy as np

from grids cimport TimeGridVel_c
from gnome.basic_types import velocity_rec


@cython.final
cdef class CyTimeGridVel(object):
    def __init__(self):
        self.timegrid = NULL

    def load_data(self, datafile, topology=None):
        cdef OSErr err
        if topology:
            err = self.timegrid.TextRead(datafile, topology)
        else:
            err = self.timegrid.TextRead(datafile, '')

        if err != 0:
            raise Exception('Failed in TextRead')

    def get_value(self, int time, location):
        cdef WorldPoint3D ref_point
        cdef VelocityRec vel
        cdef OSErr err0
        cdef char errmsg[256]

        location = np.asarray(location)
        # do the multiply by 1000000 here - this is what gnome expects
        ref_point.p.pLong = location[0]*1000000
        ref_point.p.pLat = location[1]*1000000
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
