"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import string
import os
import glob
from datetime import datetime

from colander import SchemaNode, Float, drop, Boolean

import gnome
from gnome.utilities.time_utils import sec_to_datetime
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime

from .environment import Environment
from gnome.persist import base_schema
from gnome.persist.extend_colander import FilenameSchema

from gnome.utilities.convert import tsformat
from gnome.cy_gnome.cy_ossm_time import CyTimeseries
from gnome.cy_gnome.cy_shio_time import CyShioTime


def _get_shio_yeardata_limits():
    gnome_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    yeardata_dir = os.path.join(gnome_dir, 'data', 'yeardata')
    years = [int(name[-4:]) for name in glob.glob(yeardata_dir + "/#????")]
    return (min(years), max(years))

SHIO_YEARDATA_LIMITS = _get_shio_yeardata_limits()


class TideSchema(base_schema.ObjTypeSchema):
    'Tide object schema'
    filename = FilenameSchema(
        save=True, update=False, isdatafile=True, test_equal=False
    )

    scale_factor = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    extrapolation_is_allowed = SchemaNode(Boolean(), missing=drop, read_only=True)
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class Tide(Environment):

    """
    todo: baseclass called ScaleTimeseries (or something like that)
    ScaleCurrent
    Define the tide for a spill

    Currently, this internally defines and uses the CyShioTime object, which is
    a cython wrapper around the C++ Shio object
    """
    _ref_as = 'tide'
    _schema = TideSchema

    def __init__(self,
                 filename,
                 yeardata=os.path.join(os.path.dirname(gnome.__file__),
                                       'data', 'yeardata'),
                 scale_factor=None,
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
            unit conversion - NOT IMPLEMENTED YET
        :type units=None:  (Optional) string, for example:
            'knot', 'meter per second', 'mile per hour' etc
        :param filename: path to a long wind filename from which to read
            wind data
        :param yeardata='gnome/data/yeardata/': path to yeardata used for Shio
            data.

        """
        # define locally so it is available even for OSSM files,
        # though not used by OSSM files
        super(Tide, self).__init__(**kwargs)
        self._yeardata = None
        self.filename=filename
        self.cy_obj = self._obj_to_create(filename)
        # self.yeardata = os.path.abspath( yeardata ) # set yeardata
        self.yeardata = yeardata  # set yeardata
        self.name = os.path.split(self.filename)[1]
        self.scale_factor = scale_factor if scale_factor else self.cy_obj.scale_factor


        kwargs.pop('scale_factor', None)

    @property
    def extrapolation_is_allowed(self):
        return True

    @property
    def data_start(self):
        if isinstance(self.cy_obj, CyShioTime):
            return datetime(SHIO_YEARDATA_LIMITS[0], 1, 1)
        else:
            return sec_to_datetime(self.cy_obj.get_start_time())

    @property
    def data_stop(self):
        if isinstance(self.cy_obj, CyShioTime):
            return datetime(SHIO_YEARDATA_LIMITS[1], 12, 31, 23, 59)
        else:
            return sec_to_datetime(self.cy_obj.get_end_time())

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

    scale_factor = property(lambda self:
                            self.cy_obj.scale_factor, lambda self, val:
                            setattr(self.cy_obj, 'scale_factor', val))

    def _obj_to_create(self, filename):
        """
        open file, read a few lines to determine if it is an ossm file
        or a shio file
        """
        fh = open(filename, encoding='utf-8')

        lines = [fh.readline() for i in range(4)]

        if len(lines[1]) == 0:  # should not be needed with Universal newlines, or on py3
            # look for \r for lines instead of \n
            lines = lines[0].split('\r', 4)

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
        elif len(lines[3].split(',')) == 7:
            # maybe log / display a warning that v=0 for tide file and will be
            # ignored
            # if float((lines[3].split(',')[-1]) != 0.0:
            return CyTimeseries(filename, file_format=tsformat('uv'))
        else:
            raise ValueError('This does not appear to be a valid file format '
                             'that can be read by OSSM or Shio to get '
                             'tide information')
