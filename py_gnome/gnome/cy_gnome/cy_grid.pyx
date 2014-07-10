from type_defs cimport OSErr, VelocityRec, WorldPoint3D
from libcpp cimport bool

import numpy as np

from grids cimport TimeGridVel_c, TimeGridWindRect_c


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
        ref_point.p.pLong = location[0]
        ref_point.p.pLat = location[1]
        if len(location) == 2:
            ref_point.z = 0
        else:
            ref_point.z = location[2]

        err0 = self.timegrid.SetInterval(errmsg, time)
        if err0 != 0:
            raise Exception('SetInterval error message: {0}'.format(errmsg))

        vel = self.timegrid.GetScaledPatValue(time, ref_point)
        return vel


cdef extern from *:
    TimeGridWindRect_c* dynamic_cast_ptr "dynamic_cast<TimeGridWindRect_c *>" (TimeGridVel_c *) except NULL

cdef class CyTimeGridWindRect(CyTimeGridVel):
    '''
    cython wrapper around TimeGridWindRect_c C++ class
    '''
    cdef TimeGridWindRect_c * timegridwind

    def __cinit__(self):
        self.timegrid = new TimeGridWindRect_c()
        self.timegridwind = dynamic_cast_ptr(self.timegrid)

    def __init__(self, basestring datafile, basestring topology=None):
        self.load_data(datafile, topology)
