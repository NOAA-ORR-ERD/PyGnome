"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import string
import os
import copy

from colander import SchemaNode, String, Float, drop

import gnome

from .environment import Environment
from gnome.persist import base_schema

# TODO: The name 'convert' is doubly defined as
#       unit_conversion.convert and...
#       gnome.utilities.convert
#       This will inevitably cause namespace collisions.
#       CHB-- I don't think that's a problem -- that's what namespaces are for!

from gnome.utilities.convert import tsformat


from gnome.cy_gnome.cy_ossm_time import CyTimeseries
from gnome.cy_gnome.cy_shio_time import CyShioTime


class TideSchema(base_schema.ObjTypeSchema):
    'Tide object schema'
    filename = SchemaNode(
        String(), missing=drop, save=True, update=True, isdatafile=True, test_for_eq=False
    )

    scale_factor = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )

    name = 'tide'


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
            return CyTimeseries(filename, file_format=tsformat('uv'))
        else:
            raise ValueError('This does not appear to be a valid file format '
                             'that can be read by OSSM or Shio to get '
                             'tide information')
