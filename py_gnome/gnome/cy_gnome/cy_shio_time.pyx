import numpy as np
import os
import locale

cimport numpy as cnp
from libc.string cimport memcpy

from gnome import basic_types

from .type_defs cimport *
from .utils cimport (ShioTimeValue_c,
                    #EbbFloodData, EbbFloodDataH,
                    #HighLowData, HighLowDataH,
                    _NewHandle, _GetHandleSize)
from .cy_ossm_time cimport CyOSSMTime, dynamic_cast_ptr
from .cy_helpers import filename_as_bytes


cdef class CyShioTime(CyOSSMTime):
    """
    Cython wrapper around instantiating and using ShioTimeValue_c object

    The object is declared in cy_shio_time.pxd file
    """

    def __cinit__(self):
        """ Initialize object """
        if type(self) == CyShioTime:
            self.time_dep = new ShioTimeValue_c()
            self.shio = dynamic_cast_ptr(self.time_dep)
        else:
            self.shio = NULL

    def __dealloc__(self):
        if self.time_dep is not NULL:
            del self.time_dep
            self.time_dep = NULL
            self.shio = NULL

    def __init__(self,
                 filename,
                 scale_factor=1,
                 daylight_savings_off=False,  # let shio figure out by default
                 yeardata=None):
        """
        Invoke super and call CyOSSMTime.__init__ with filename, timeseries,
        scale_factor. The file_format is always set to 0 since shio is neither
        in 'uv', nor 'r-theta' format and Shio does not use this for data read
        """
        super(CyShioTime, self).__init__(filename, 0, scale_factor)

        # base class will set file_format to basic_types.ts_format.uv
        self.shio.daylight_savings_off = daylight_savings_off
        if not yeardata:
            yeardata = os.path.join(os.path.dirname(basic_types.__file__),
                                    'data/yeardata/')
        self.set_shio_yeardata_path(yeardata)
        self._yeardata_path = yeardata

    def __reduce__(self):
        return (
            CyShioTime,
            (
                self._cy_filename,
                self.time_dep.fScaleFactor,
                self.shio.daylight_savings_off,
                self._yeardata_path
            )
        )

    def set_shio_yeardata_path(self, yeardata_path_):
        """
        .. function::set_shio_yeardata_path
        C++ expects a trailing slash at the end of yeardata_path,
        this is explicitly added here
        """
        cdef OSErr err
        cdef bytes yeardata_path

        # path separator as a bytes object
        cdef bytes bsep = os.sep.encode("ASCII")
        yeardata_path = filename_as_bytes(yeardata_path_)

        if os.path.exists(yeardata_path):
            if yeardata_path[-1] != bsep:
                yeardata_path = os.path.normpath(yeardata_path) + bsep

            # implicit conversion from bytes to char *
            err = self.shio.SetYearDataPath(yeardata_path)
            if err != 0:
                raise ValueError("Path could not be correctly be set by "
                                 "ShioTimeValue_c.SetYearDataPath(...)")
        else:
            raise IOError("No such file: " + yeardata_path_)

    def _read_time_values(self, filename):
        '''
        Call Shio's ReadTimeValues which has different signature than base
        class. Also set _file_format to 0 since it isn't in 'uv' or 'r-theta'
        format

        :param filename: path of the file
        :type filename: PathLike
        '''
        cdef bytes file_ = filename_as_bytes(filename)
        err = self.shio.ReadShioValues(file_)
        self._raise_errors(err)

    property daylight_savings_off:
        def __get__(self):
            return self.shio.daylight_savings_off

        def __set__(self, bool value):
            self.shio.daylight_savings_off = value

    property yeardata:
        def __get__(self):
            """ get path of yeardata files """
            return os.fsdecode(self.shio.fYearDataPath)

        def __set__(self, value):
            """ set path of yeardata files """
            # TODO: figure out how to change fYearDataPath directly
            self.set_shio_yeardata_path(value)

    property station_type:
        def __get__(self):
            """
            station type: 'C', 'H', 'P' - not sure what these refer to yet?
            """
            cdef bytes sType
            sType = self.shio.fStationType
            return sType.decode('ascii')

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated
        """
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                '{0.path_filename!r}, '
                'daylight_savings_off={0.daylight_savings_off}, '
                'scale_factor={0.scale_factor}, '
                'yeardata={0.yeardata!r}'
                ')'.format(self))

    def __str__(self):
        """Return string representation of this object"""
        """info = {'Long': round(g_wp[0]['long'], 2),
                   'Lat': round( g_wp[0]['lat'], 2),
                   'StationName': sName, 'StationType': sType,
                   'DaylightSavingsOff': self.shio.daylight_savings_off}
        """

        info = ("CyShioTime object - Info read from file:\n"
                "  File: {1.path_filename} \n"
                "  StationName : {0[StationName]},"
                "  StationType : {0[StationType]}\n"
                "  (Long, Lat) : ({0[Long]}, {0[Lat]})\n"
                "  DaylightSavingsOff : {0[DaylightSavingsOff]}"
                "".format(self.get_info(), self))

        return info

    def __eq(self, CyShioTime other):
        attrs = ('filename', 'daylight_savings_off', 'scale_factor',
                 'station', 'station_type', 'station_location',
                 'yeardata')
        return all([getattr(self, a) == getattr(other, a) for a in attrs])

    def __richcmp__(self, CyShioTime other, int cmp):
        if cmp not in (2, 3):
            raise NotImplemented('CyOSSMTime does not support '
                                 'this type of comparison.')

        if cmp == 2:
            return self.__eq(other)
        elif cmp == 3:
            return not self.__eq(other)

#==============================================================================
#     def get_ebb_flood(self, modelTime):
#         """
#         Return ebb flood data for specified modelTime
#         """
#         # initialize self.shio.fEbbFloodDataHdl for specified duration
#         self.get_time_value(modelTime)
#
#         cdef short tmp_size = sizeof(EbbFloodData)
#         cdef cnp.ndarray[EbbFloodData, ndim = 1] ebb_flood
#
#         if self.shio.fStationType == 'C':
#             # allocate memory and copy it over
#             sz = _GetHandleSize(<Handle>self.shio.fEbbFloodDataHdl)
#             ebb_flood = np.empty((sz // tmp_size,),
#                                  dtype=basic_types.ebb_flood_data)
#             memcpy(&ebb_flood[0], self.shio.fEbbFloodDataHdl[0], sz)
#             return ebb_flood
#         else:
#             return 0
#
#     def get_high_low(self, modelTime):
#         """
#         Return high and low tide data for specified modelTime
#         """
#         # initialize self.shio.fEbbFloodDataHdl for specified duration
#         self.get_time_value(modelTime)
#
#         cdef short tmp_size = sizeof(HighLowData)
#         cdef cnp.ndarray[HighLowData, ndim = 1] high_low
#
#         if self.shio.fStationType == 'H':
#             # initialize self.shio.fEbbFloodDataHdl for specified duration
#             self.get_time_value(modelTime)
#
#             # allocate memory and copy it over
#             sz = _GetHandleSize(<Handle>self.shio.fHighLowDataHdl)
#             high_low = np.empty((sz // tmp_size,),
#                                 dtype=basic_types.ebb_flood_data)
#             memcpy(&high_low[0], self.shio.fHighLowDataHdl[0], sz)
#             return high_low
#         else:
#             return 0
#==============================================================================
