"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import string
import os
import copy
import datetime

import numpy
np = numpy

from colander import SchemaNode, String, Float, drop

import gnome
from gnome.basic_types import time_value_pair

from .environment import Environment
from gnome.persist import validators, base_schema
from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue2dArraySchema)

# TODO: The name 'convert' is doubly defined as
#       hazpy.unit_conversion.convert and...
#       gnome.utilities.convert
#       This will inevitably cause namespace collisions.

from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.convert import (to_time_value_pair,
                                     to_datetime_value_2d,
                                     tsformat)

from gnome.utilities.serializable import Serializable, Field

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_shio_time import CyShioTime


#==============================================================================
# class UVTuple(DefaultTupleSchema):
#     u = SchemaNode(Float())
#     v = SchemaNode(Float())
# 
# 
# class TimeSeriesTuple(DefaultTupleSchema):
#     '''
#     Schema for each tuple in WindTimeSeries list
#     '''
#     datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
#                           default=base_schema.now,
#                           validator=validators.convertible_to_seconds)
#     uv = UVTuple()
# 
# 
# class TimeSeriesSchema(DatetimeValue2dArraySchema):
#     '''
#     Schema for list of Wind tuples, to make the wind timeseries
#     '''
#     value = TimeSeriesTuple(default=(datetime.datetime.now(), 0, 0))
# 
#     def validator(self, node, cstruct):
#         '''
#         validate wind timeseries numpy array
#         '''
#         validators.no_duplicate_datetime(node, cstruct)
#         validators.ascending_datetime(node, cstruct)
#==============================================================================


class TideSchema(base_schema.ObjType):
    'Tide object schema'
    filename = SchemaNode(String(), missing=drop)

    #timeseries = TimeSeriesSchema(missing=drop)
    name = 'tide'


class Tide(Environment, Serializable):

    """
    todo: baseclass called ScaleTimeseries (or something like that)
    ScaleCurrent
    Define the tide for a spill

    Currently, this internally defines and uses the CyShioTime object, which is
    a cython wrapper around the C++ Shio object
    """

    _state = copy.deepcopy(Environment._state)
    _schema = TideSchema

    # add 'filename' as a Field object
    _state.add_field(Field('filename', save=True, read=True, isdatafile=True,
                           test_for_eq=False))

    def __init__(self,
                 filename,
                 yeardata=os.path.join(os.path.dirname(gnome.__file__),
                                       'data', 'yeardata'),
                 **kwargs):
        """
        Tide information can be obtained from a filename or set as a
        timeseries (timeseries is NOT TESTED YET)

        It requires one of the following to initialize:

              1. 'timeseries' assumed to be in 'uv' format
                 (NOT TESTED/IMPLEMENTED OR USED YET)
              2. a 'filename' containing a header that defines units amongst
                 other meta data

        :param timeseries: numpy array containing tide data
        :type timeseries: numpy.ndarray with dtype=datetime_value_1d
        :param units: units associated with the timeseries data. If 'filename'
            is given, then units are read in from the filename.
            unit_conversion - NOT IMPLEMENTED YET
        :type units=None:  (Optional) string, for example:
            'knot', 'meter per second', 'mile per hour' etc
        :param filename: path to a long wind filename from which to read
            wind data
        :param yeardata='gnome/data/yeardata/': path to yeardata used for Shio
            data.

        """
        # define locally so it is available even for OSSM files,
        # though not used by OSSM files
        self._yeardata = None
        self.cy_obj = self._obj_to_create(filename)
        # self.yeardata = os.path.abspath( yeardata ) # set yeardata
        self.yeardata = yeardata  # set yeardata
        self.name = kwargs.pop('name', os.path.split(self.filename)[1])

        super(Tide, self).__init__(**kwargs)

    @property
    def yeardata(self):
        return self._yeardata

    @yeardata.setter
    def yeardata(self, value):
        """
        only relevant if underlying cy_obj is CyShioTime
        """
        if not os.path.exists(value):
            raise IOError('Path to yeardata files does not exist: '
                          '{0}'.format(value))

        # set private variable and also shio object's yeardata path
        self._yeardata = value

        if isinstance(self.cy_obj, CyShioTime):
            self.cy_obj.set_shio_yeardata_path(value)

    filename = property(lambda self: (self.cy_obj.filename, None
                                      )[self.cy_obj.filename == ''])

    def _obj_to_create(self, filename):
        """
        open file, read a few lines to determine if it is an ossm file
        or a shio file
        """
        # mode 'U' means universal newline support
        fh = open(filename, 'rU')

        lines = [fh.readline() for i in range(4)]

        if len(lines[1]) == 0:
            # look for \r for lines instead of \n
            lines = string.split(lines[0], '\r', 4)

        if len(lines[1]) == 0:
            # if this is still 0, then throw an error!
            raise ValueError('This does not appear to be a valid file format '
                             'that can be read by OSSM or Shio to get '
                             'tide information')

        # look for following keywords to determine if it is a Shio or OSSM file
        shio_file = ['[StationInfo]', 'Type=', 'Name=', 'Latitude=']

        if all([shio_file[i] == (lines[i])[:len(shio_file[i])]
                for i in range(4)]):
            return CyShioTime(filename)
        elif len(string.split(lines[3], ',')) == 7:
            # maybe log / display a warning that v=0 for tide file and will be
            # ignored
            # if float( string.split(lines[3],',')[-1]) != 0.0:
            return CyOSSMTime(filename, file_contains=tsformat('uv'))
        else:
            raise ValueError('This does not appear to be a valid file format '
                             'that can be read by OSSM or Shio to get '
                             'tide information')
