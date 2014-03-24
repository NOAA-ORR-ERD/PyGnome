"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import string
import os
import copy

from colander import (SchemaNode,
                      drop,
                      String)

from .environment import Environment
from gnome.persist import base_schema

import gnome

# TODO: The name 'convert' is doubly defined as
#       hazpy.unit_conversion.convert and...
#       gnome.utilities.convert
#       This will inevitably cause namespace collisions.

from gnome.utilities import convert, serializable

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_shio_time import CyShioTime


class TideSchema(base_schema.ObjType):
    'Tide object schema'
    filename = SchemaNode(String(), missing=drop)
    yeardata = SchemaNode(String())
    name = 'tide'


class Tide(Environment, serializable.Serializable):

    """
    todo: baseclass called ScaleTimeseries (or something like that)
    ScaleCurrent
    Define the tide for a spill

    Currently, this internally defines and uses the CyShioTime object, which is
    a cython wrapper around the C++ Shio object
    """

    _state = copy.deepcopy(Environment._state)

    # no need to copy parent's _state in this case

    _state.add(create=['yeardata'], update=['yeardata'])

    # add 'filename' as a Field object

    _state.add_field(serializable.Field('filename', isdatafile=True,
                    create=True, read=True, test_for_eq=False))
    _schema = TideSchema

    def __init__(
        self,
        filename=None,
        timeseries=None,
        yeardata=os.path.join(os.path.dirname(gnome.__file__), 'data',
                              'yeardata'),
        **kwargs
        ):
        """
        Tide information can be obtained from a filename or set as a
        timeseries (timeseries is NOT TESTED YET)

        Invokes super(Tides,self).__init__(\*\*kwargs) for parent class
        initialization

        It requires one of the following to initialize:
              1. 'timeseries' assumed to be in 'uv' format
                 (NOT TESTED/IMPLEMENTED OR USED YET)
              2. a 'filename' containing a header that defines units amongst
                 other meta data

        :param timeseries: numpy array containing datetime_value_2d,
                           ts_format is always 'uv'
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param units: units associated with the timeseries data. If 'filename'
                      is given, then units are read in from the filename.
                      unit_conversion - NOT IMPLEMENTED YET
        :type units:  (Optional) string, for example:
                        'knot', 'meter per second', 'mile per hour' etc
                      Default is None for now

        :param filename: path to a long wind filename from which to read
                         wind data

        :param yeardata: (Optional) path to yeardata used for Shio data
                         filenames. Default location is gnome/data/yeardata/

        Remaining kwargs ('id' if present) are passed onto Environment's
        __init__ using super.
        See base class documentation for remaining valid kwargs.
        """

        # define locally so it is available even for OSSM files,
        # though not used by OSSM files

        self._yeardata = None

        if timeseries is None and filename is None:
            raise ValueError('Either provide timeseries or a valid filename'
                ' containing Tide data')

        if timeseries is not None:

#            if units is None:
#                raise ValueError("Provide valid units as string or unicode " \
#                                 "for timeseries")

            # will probably need to move this function out
            # self._check_timeseries(timeseries, units)

            # data_format is checked during conversion

            time_value_pair = convert.to_time_value_pair(timeseries,
                    convert.tsformat('uv'))

            # this has same scope as CyWindMover object

            self.cy_obj = CyOSSMTime(timeseries=time_value_pair)

            # not sure what these should be

            self._user_units = kwargs.pop('units', None)
        else:

            # self.filename = os.path.abspath( filename)

            self.cy_obj = self._obj_to_create(filename)

            # self.yeardata = os.path.abspath( yeardata ) # set yeardata

            self.yeardata = yeardata  # set yeardata

        super(Tide, self).__init__(**kwargs)

    @property
    def yeardata(self):
        return self._yeardata

    @yeardata.setter
    def yeardata(self, value):
        """ only relevant if underlying cy_obj is CyShioTime"""

        if not os.path.exists(value):
            raise IOError('Path to yeardata files does not exist:'
                ' {0}'.format(value))

        # set private variable and also shio object's yeardata path

        self._yeardata = value

        if isinstance(self.cy_obj, CyShioTime):
            self.cy_obj.set_shio_yeardata_path(value)

    filename = property(lambda self: (self.cy_obj.filename,
                        None)[self.cy_obj.filename == ''])

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

            raise ValueError('This does not appear to be a valid file format'
                ' that can be read by OSSM or Shio to get tide information')

        # look for following keywords to determine if it is a Shio or OSSM file

        shio_file = ['[StationInfo]', 'Type=', 'Name=', 'Latitude=']

        if all([shio_file[i] == (lines[i])[:len(shio_file[i])] for i in
               range(4)]):
            return CyShioTime(filename)
        elif len(string.split(lines[3], ',')) == 7:

            # maybe log / display a warning that v=0 for tide file and will be
            # ignored
            # if float( string.split(lines[3],',')[-1]) != 0.0:

            return CyOSSMTime(filename,
                              file_contains=convert.tsformat('uv'))
        else:
            raise ValueError('This does not appear to be a valid file format'
                ' that can be read by OSSM or Shio to get tide information')
