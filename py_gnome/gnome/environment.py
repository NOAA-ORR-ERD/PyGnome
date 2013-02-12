"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import datetime
import string

import numpy as np
from hazpy import unit_conversion

from gnome import basic_types, GnomeObject
from gnome.utilities import transforms, time_utils, convert
from gnome.cy_gnome import cy_ossm_time, cy_shio_time

class Wind(GnomeObject):
    """
    Defines the Wind conditions for a spill
    """
    def __init__(self, 
                 timeseries=None, 
                 file=None,
                 format='r-theta',
                 units=None):
        """
        Initializes a wind object. It requires a numpy array containing 
        gnome.basic_types.datetime_value_2d which defines the wind velocity
        
        :param timeseries: (Required) numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param file: path to a long wind file from which to read wind data
        :param format: default timeseries format is magnitude_direction
        :type format: string "r-theta" or "uv". 
                      Converts string to integer defined by gnome.basic_types.ts_format.*
        :param units: units associated with the timeseries data. If 'file' is given, then units are read in from the file. 
                      These units must be valid as defined in the hazpy unit_conversion module: 
                      unit_conversion.GetUnitNames('Velocity') 
        :type units:  string, for example: 'knot', 'meter per second', 'mile per hour' etc
        """
        
        if( timeseries is None and file is None):
            raise ValueError("Either provide timeseries or a valid long file")
        
        if( timeseries is not None):
            if units is None:
                raise ValueError("Provide valid units as string or unicode for timeseries")
            
            self._check_timeseries(timeseries, units)
            
            timeseries['value'] = self._convert_units(timeseries['value'], format, units, 'meter per second')
            time_value_pair = convert.to_time_value_pair(timeseries, format)   # ts_format is checked during conversion
                
            self.ossm = cy_ossm_time.CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            self._user_units = units
        else:
            ts_format = convert.tsformat(format)
            self.ossm = cy_ossm_time.CyOSSMTime(path=file,file_contains=ts_format)
            self._user_units = self.ossm.user_units
        
    def _convert_units(self, data, ts_format, from_unit, to_unit):
        """
        Private method to convert units for the 'value' stored in the date/time value pair
        """
        if from_unit != to_unit:
            data[:,0]  = unit_conversion.convert('Velocity', from_unit, to_unit, data[:,0])
            if ts_format == basic_types.ts_format.uv:
                data[:,1]  = unit_conversion.convert('Velocity', from_unit, to_unit, data[:,1])
        
        return data
    
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
        unique = np.unique( timeseries)
        if len( unique) != len(timeseries):
            raise ValueError('timeseries must contain unique entries. Number of duplicate entries ' + str(len(timeseries)-len(unique) ) )
        
    def __repr__(self):
        """
        Return an unambiguous representation of this `Wind object` so it can be recreated
        
        This timeseries are not output. eval(repr(wind)) does not work for this object and the timeseries could be long
        so only the syntax for obtaining the timeseries is given in repr
        """
        return "Wind( timeseries=Wind.get_timeseries(basic_types.ts_format.uv), ts_format=basic_types.ts_format.uv)" \
    
    def __str__(self):
        """
        Return string representation of this object
        """
        return "Wind Object"
    
    user_units = property( lambda self: self._user_units)   
    
    def get_timeseries(self, datetime=None, units=None, format='r-theta'):
        """
        returns the timeseries in the requested format. If datetime=None, then the original timeseries
        that was entered is returned. If datetime is a list containing datetime objects, then the
        wind value for each of those date times is determined by the underlying CyOSSMTime object and
        the timeseries is returned.  

        The output format is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime objects for which the value is desired
        :type datetime: datetime object
        :param units: [optional] outputs data in these units. Default is to output data in user_units
        :type units: string. Uses the hazpy.unit_conversion module
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
            datetimeval['value'] = self._convert_units(datetimeval['value'], format, 'meter per second', self.user_units)
            
        return datetimeval
    
    def set_timeseries(self, datetime_value_2d, units, format='r-theta'):
        """
        sets the timeseries of the Wind object to the new value given by a numpy array. 
        The format for the input data defaults to 
        basic_types.format.magnitude_direction but can be changed by the user
        
        :param datetime_value_2d: timeseries of wind data defined in a numpy array
        :type datetime_value_2d: numpy array of dtype basic_types.datetime_value_2d
        :param user_units: XXX user units
        :param format: output format for the times series; as defined by basic_types.format.
        :type format: either string or integer value defined by basic_types.format.* (see cy_basic_types.pyx)
        """
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
        
        :param timeseries: (Required) numpy array containing datetime_value_2d, ts_format is always uv
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param file: path to a long wind file from which to read wind data
        :param units: units associated with the timeseries data. If 'file' is given, then units are read in from the file. 
                      unit_conversion - NOT IMPLEMENTED YET
        :type units:  string, for example: 'knot', 'meter per second', 'mile per hour' etc
        """
        if( timeseries is None and file is None):
            raise ValueError("Either provide timeseries or a valid file containing Tide data")
        
        if( timeseries is not None):
            if units is None:
                raise ValueError("Provide valid units as string or unicode for timeseries")
            
            #self._check_timeseries(timeseries, units)    # will probably need to move this function out
            
            time_value_pair = convert.to_time_value_pair(timeseries, basic_types.data_format.wind_uv)   # data_format is checked during conversion
                
            self.cy_obj = cy_ossm_time.CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            self._user_units = units
        else:
            self.cy_obj = self._obj_to_create(file)
            
            
    def _obj_to_create(self, file_):
        """
        open file, read a few lines to determine if it is an ossm file or a shio file
        """
        fh = open(file_, 'r')
        lines = [fh.readline() for i in range(4)]   # depends on line endings - this does not work for '\r'
        
        if len(lines[1]) == 0:
            # look for \r for lines instead of \n
            lines = string.split(lines[0],'\r',4) 
            
        if len(lines[1]) == 0:
            # if this is still 0, then throw an error!
            raise ValueError("This does not appear to be a valid file format that can be read by OSSM or Shio to get tide information")
        
        shio_file = ['[StationInfo]', 'Type=', 'Name=', 'Latitude=']
        
        if all( [shio_file[i] == lines[i][:len(shio_file[i])] for i in range(4)]):
            return cy_shio_time.CyShioTime(file_)
        
        elif len(string.split(lines[3],',')) == 7:
            #if float( string.split(lines[3],',')[-1]) != 0.0:    # maybe log / display a warning that v=0 for tide file and will be ignored
            return cy_ossm_time.CyOSSMTime(file_, data_format=basic_types.data_format.uv)
        else:
            raise ValueError("This does not appear to be a valid file format that can be read by OSSM or Shio to get tide information") 
