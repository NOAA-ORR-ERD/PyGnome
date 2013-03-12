"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import datetime
import os
import copy

import numpy as np

from gnome import basic_types, GnomeObject
from gnome.utilities import transforms, time_utils, convert, serializable
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from hazpy import unit_conversion

class Wind( GnomeObject, serializable.Serializable):
    """
    Defines the Wind conditions for a spill
    """
    # removed 'id' from list below. id, filename and units cannot be updated - read only properties
    _update = [ 'name',
                'latitude',
                'longitude',
                'description',
                'source_id',
                'source_type',
                'updated_at',
                'timeseries',
                'units']    # default units for input/output data
    _create = []
    _create.extend(_update)
    
    state = copy.deepcopy(serializable.Serializable.state)
    state.add(read  =['filename'],
              create=_create,
              update=_update)   # no need to copy parent's state in tis case

    # list of valid velocity units for timeseries
    valid_vel_units = unit_conversion.GetUnitNames('Velocity')
    valid_vel_units.extend([unit_conversion.GetUnitAbbreviation('Velocity',unit_) for unit_ in unit_conversion.GetUnitNames('Velocity')])

    def __init__(self, **kwargs):
        """
        Initializes a wind object. It only takes keyword arguments as input, these
        are defined below. It requires one of the following to initialize:
              1. 'timeseries' along with 'units' or
              2. a 'file' containing a header that defines units amongst other meta data
               
        All other keywords are optional.
        
        :param timeseries: (Required) numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        
        :param file: path to a long wind file from which to read wind data
        :param units: units associated with the timeseries data. If 'file' is given, then units are read in from the file. 
                      get_timeseries() will use these as default units to output data, unless user specifies otherwise.
                      These units must be valid as defined in the hazpy unit_conversion module: 
                      unit_conversion.GetUnitNames('Velocity') 
        :type units:  string, for example: 'knot', 'meter per second', 'mile per hour' etc. Default units for input/output timeseries data
        
        :param format: (Optional) default timeseries format is magnitude direction: 'r-theta'
        :type format: string 'r-theta' or 'uv'. Converts string to integer defined by gnome.basic_types.ts_format.*
        
        :param name: (Optional) human readable string for wind object name. Default is filename if data is from file or "Wind Object"
        
        :param source_type: (Optional) Default is undefined, but can be one of the following: ['buoy', 'manual', 'undefined', 'file', 'nws']
                            If data is read from file, then it is 'file'
                            
        :param latitude: (Optional) latitude of station or location where wind data is obtained from NWS
        :param longitude: (Optional) longitude of station or location where wind data is obtained from NWS
        
        :param filename: (Optional) timeseries could have come from a file and user may want to store that as meta data
        """
        
        if 'timeseries' in kwargs and 'file' in kwargs:
            raise TypeError("Cannot instantiate Wind object with both timeseries and file as input")
        
        if 'timeseries' not in kwargs and 'file' not in kwargs:
            raise TypeError("Either provide a timeseries or a wind file with a header, containing wind data")
        
        # default lat/long - can these be set from reading data in the file?
        self.longitude = None
        self.latitude = None
        
        # format of data 'uv' or 'r-theta'. Default is 'r-theta'
        format = kwargs.pop('format', 'r-theta')
        self.description = kwargs.pop('description','Wind Object')
        if 'timeseries' in kwargs:
            if 'units' not in kwargs:
                raise TypeError("Provide 'units' argument with the 'timeseries' input")
            timeseries = kwargs.pop('timeseries')
            units = kwargs.pop('units')
            
            self._check_units(units)
            self._check_timeseries(timeseries, units)
            
            timeseries['value'] = self._convert_units(timeseries['value'], format, units, 'meter per second')
            time_value_pair = convert.to_time_value_pair(timeseries, format)   # ts_format is checked during conversion
                
            self.ossm = CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            self._user_units = units    # do not set ossm.user_units since that only has a subset of possible units
            
            self.name = kwargs.pop('name','Wind Object')
            self.source_type= kwargs.pop('source_type') if kwargs.get('source_type') in basic_types.wind_datasource._attr else 'undefined'
            
        else:
            ts_format = convert.tsformat(format)
            self.ossm = CyOSSMTime(file=kwargs.pop("file"),file_contains=ts_format)
            self._user_units = self.ossm.user_units
            
            self.name = kwargs.pop('name',os.path.split(self.ossm.filename)[1])
            self.source_type = 'file'   # this must be file
        
        # For default: if read from file and filename exists, then use last modified time of file
        #              else default to datetime.datetime.now
        # not sure if this should be datetime or string
        self.updated_at = kwargs.pop('updated_at', 
                                     time_utils.sec_to_date( os.path.getmtime(self.ossm.filename)) \
                                     if self.ossm.filename else datetime.datetime.now() )
        self.source_id = kwargs.pop('source_id','undefined')
        self.longitude = kwargs.pop('longitude',self.longitude)
        self.latitude = kwargs.pop('latitude',self.latitude)
        
        
    def _convert_units(self, data, ts_format, from_unit, to_unit):
        """
        Private method to convert units for the 'value' stored in the date/time value pair
        """
        if from_unit != to_unit:
            data[:,0]  = unit_conversion.convert('Velocity', from_unit, to_unit, data[:,0])
            if ts_format == basic_types.ts_format.uv:
                data[:,1]  = unit_conversion.convert('Velocity', from_unit, to_unit, data[:,1])
        
        return data
    
    def _check_units(self, units):
        """
        Checks the user provided units are in list Wind.valid_vel_units 
        """
        if units not in Wind.valid_vel_units:
            raise unit_conversion.InvalidUnitError('Velocity units must be from following list to be valid: {0}'.format(Wind.valid_vel_units))
        
    def _check_timeseries(self, timeseries, units):
        """
        Run some checks to make sure timeseries is valid
        """
        try:
            if( timeseries.dtype is not basic_types.datetime_value_2d):
                # Both 'is' or '==' work in this case. There is only one instance of basic_types.datetime_value_2d
                # Maybe in future we can consider working with a list, but that's a bit more cumbersome for different dtypes
                raise ValueError("timeseries must be a numpy array containing basic_types.datetime_value_2d dtype")
        
        except AttributeError as err:
            raise AttributeError("timeseries is not a numpy array. " + err.message)
        
        # check to make sure the time values are in ascending order
        if np.any( timeseries['time'][np.argsort( timeseries['time'])] != timeseries['time']):
            raise ValueError('timeseries are not in ascending order. The datetime values in the array must be in ascending order')
        
        # check for duplicate entries
        unique = np.unique( timeseries['time'])
        if len( unique) != len(timeseries['time']):
            raise ValueError('timeseries must contain unique time entries. Number of duplicate entries ' + str(len(timeseries)-len(unique) ) )
        
    def __repr__(self):
        """
        Return an unambiguous representation of this `Wind object` so it can be recreated
        
        This timeseries are not output. eval(repr(wind)) does not work for this object and the timeseries could be long
        so only the syntax for obtaining the timeseries is given in repr
        """
        return "Wind( timeseries=Wind.get_timeseries('uv'), format='uv')" \
    
    def __str__(self):
        """
        Return string representation of this object
        """
        return "Wind Object"
    
    
    def __eq__(self,other):
        # since this has numpy array - need to compare that as well
        # By default, tolerance for comparison is atol=1e-10, rtol=0
        # persisting data requires unit conversions and finite precision, both of which will
        # introduce a difference between two objects
        check = super(Wind,self).__eq__(other)
        
        if check:
            if (self.timeseries['time'] != other.timeseries['time']).all():
                return False
            else:
                return np.allclose(self.timeseries['value'], other.timeseries['value'], 1e-10, 0)
        
        # user_units is also not part of state.create list so do explicit check here
        #if self.user_units != other.user_units:
        #    return False
                
        return check
        
    #user_units = property( lambda self: self._user_units)
    
    @property
    def units(self):
        return self._user_units
    
    @units.setter
    def units(self, value):
        """
        User can set default units for input/output data
        
        These are given as string and must be one of the units defined in
        unit_conversion.GetUnitNames('Velocity') or one of the associated abbreviations
        unit_conversion.GetUnitAbbreviation()
        """
        self._check_units(value)
        self._user_units = value
    
    filename = property( lambda self: self.ossm.filename)
    timeseries = property( lambda self: self.get_timeseries(),
                           lambda self, val: self.set_timeseries(val, units=self.units) )
    
    def get_timeseries(self, datetime=None, units=None, format='r-theta'):
        """
        returns the timeseries in the requested format. If datetime=None, then the original timeseries
        that was entered is returned. If datetime is a list containing datetime objects, then the
        wind value for each of those date times is determined by the underlying CyOSSMTime object and
        the timeseries is returned.  

        The output format is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime objects for which the value is desired
        :type datetime: datetime object
        :param units: [optional] outputs data in these units. Default is to output data in units
        :type units: string. Uses the hazpy.unit_conversion module. hazpy.unit_conversion throws error for invalid units
        :param format: output format for the times series: either 'r-theta' or 'uv'
        :type format: either string or integer value defined by basic_types.ts_format.* (see cy_basic_types.pyx)

        :returns: numpy array containing dtype=basic_types.datetime_value_2d. Contains user specified datetime
            and the corresponding values in user specified ts_format
        """
        if datetime is None:
            datetimeval = convert.to_datetime_value_2d(self.ossm.timeseries, format)
        else:
            datetime = np.asarray(datetime, dtype='datetime64[s]').reshape(-1,)
            timeval = np.zeros((len(datetime),),dtype=basic_types.time_value_pair)
            timeval['time'] = time_utils.date_to_sec(datetime)
            timeval['value'] = self.ossm.get_time_value(timeval['time'])
            datetimeval = convert.to_datetime_value_2d(timeval, format)
        
        if units is not None:
            datetimeval['value'] = self._convert_units(datetimeval['value'], format, 'meter per second', units)
        else:
            datetimeval['value'] = self._convert_units(datetimeval['value'], format, 'meter per second', self.units)
            
        return datetimeval
    
    def set_timeseries(self, datetime_value_2d, units, format='r-theta'):
        """
        sets the timeseries of the Wind object to the new value given by a numpy array. 
        The format for the input data defaults to 
        basic_types.format.magnitude_direction but can be changed by the user
        
        :param datetime_value_2d: timeseries of wind data defined in a numpy array
        :type datetime_value_2d: numpy array of dtype basic_types.datetime_value_2d
        :param units: units associated with the data. Valid units defined in Wind.valid_vel_units list
        :param format: output format for the times series; as defined by basic_types.format.
        :type format: either string or integer value defined by basic_types.format.* (see cy_basic_types.pyx)
        """
        self._check_units(units)
        self._check_timeseries(datetime_value_2d, units)
        datetime_value_2d['value'] = self._convert_units(datetime_value_2d['value'], format, units, 'meter per second')
        
        timeval = convert.to_time_value_pair(datetime_value_2d, format)
        self.ossm.timeseries = timeval
    

def ConstantWind(speed, direction, units='m/s'):
    """
    utility to create a constant wind "timeseries"
    
    :param speed: speed of wind 
    :param direction: direction -- degrees True, direction wind is from( degrees True )
    :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s", "cm/s", etc.
    """
    wind_vel = np.zeros((1,), dtype=basic_types.datetime_value_2d)
    wind_vel['time'][0] = datetime.datetime.now() # jsut to have a time 
    wind_vel['value'][0] = (speed, direction)
    
    return Wind(timeseries=wind_vel,
                format='r-theta',
                units=units)

        
class Tides(GnomeObject):
    """
    Define the tides for a spill
    
    Currently, this internally defines and uses the CyShioTime object, which is
    a cython wrapper around the C++ Shio object
    """
    def __init__(self,
                 timeseries=None,
                 file=None,
                 units=None):
        """
        Tide information can be obtained from a file or set as a timeseries
        
        :param timeseries: (Required) numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param file: path to a long wind file from which to read wind data
        :param units: units associated with the timeseries data. If 'file' is given, then units are read in from the file. 
                      unit_conversion - NOT IMPLEMENTED YET
        :type units:  string, for example: 'knot', 'meter per second', 'mile per hour' etc
        """
        self.ossm_format = basic_types.ts_format.uv
        if( timeseries is None and file is None):
            raise ValueError("Either provide timeseries or a valid file containing Tide data")
        
        if( timeseries is not None):
            if units is None:
                raise ValueError("Provide valid units as string or unicode for timeseries")
            
            self._check_timeseries(timeseries, units)
            
            timeseries['value'] = self._convert_units(timeseries['value'], ts_format, units, 'meter per second')
            time_value_pair = convert.to_time_value_pair(timeseries, ts_format)   # ts_format is checked during conversion
                
            self.ossm = CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            self._user_units = units
        else:
            self.ossm = CyOSSMTime(path=file,file_contains=ts_format)
            self._user_units = self.ossm.user_units
