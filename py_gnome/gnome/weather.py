"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
from gnome import basic_types
from gnome.helpers import convert
from gnome.utilities import transforms, time_utils
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime

import numpy as np

class Wind(object):
    """
    Defines the Wind conditions for a spill
    """
    def __init__(self, 
                 timeseries=None, 
                 file=None,
                 data_format=basic_types.data_format.magnitude_direction,
                 units=basic_types.velocity_units.meters_per_sec):
        """
        Initializes a wind object. It requires a numpy array containing 
        gnome.basic_types.datetime_value_2d which defines the wind velocity
        
        :param timeseries: (Required) numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param file: path to a long wind file from which to read wind data
        :param data_format: default data_format, either magnitude_direction or wind_uv
        :type data_format: integer defined by gnome.basic_types.data_format.*
        :param units: units for the timeseries. If 'file' is given, then units are read in from the file. 
                      CURRENTLY, JUST A PLACE HOLDER. THIS IS NOT IMPLEMENTED YET
        :type units: Currently, using basic_types.velocity_units.meters_per_sec .. but still working on this
        """
        
        if( timeseries is None and file is None):
            raise ValueError("Either provide timeseries or a valid long file")
        
        if( timeseries is not None):
            try:
                if( timeseries.dtype is not basic_types.datetime_value_2d):
                    # Both 'is' or '==' work in this case. There is only one instance of basic_types.datetime_value_2d
                    # Maybe in future we can consider working with a list, but that's a bit more cumbersome for different dtypes
                    raise ValueError("timeseries must be a numpy array containing basic_types.datetime_value_2d dtype")
            
            except AttributeError as err:
                raise AttributeError("timeseries is not a numpy array. " + err.message)
            
            time_value_pair = convert.to_time_value_pair(timeseries, data_format)
            self.ossm = CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            
        else:
            self.ossm = CyOSSMTime(path=file,file_contains=data_format)
        
    def __repr__(self):
       """
       Return an unambiguous representation of this `Wind object` so it can be recreated
       
       This timeseries are not output. eval(repr(wind)) does not work for this object and the timeseries could be long
       so only the syntax for obtaining the timeseries is given in repr
       """
       return "Wind( timeseries=Wind.get_timeseries(basic_types.data_format.wind_uv), data_format=basic_types.data_format.wind_uv)" \
    
    def __str__(self):
        """
        Return string representation of this object
        """
        return "Wind Object"
    
    @property
    def id(self):
        """
        Return an ID value for this object

        This method uses Python's builtin `id()` function to identify the
        object. Override it for more exotic forms of identification.

        :return: the integer ID returned by id() for this object
        """
        return id(self)
    
    
    def get_timeseries(self, data_format, datetime=None):
        """
        returns the timeseries in the requested format. If datetime=None, then the original timeseries
        that was entered is returned. If datetime is a list containing datetime objects, then the
        wind value for each of those date times is determined by the underlying CyOSSMTime object and
        the timeseries is returned.  
        
        The output data_format is defined by the basic_types.data_format
        
        :param data_format: output format for the times series; as defined by basic_types.data_format.
        :type data_format: integer value defined by basic_types.data_format.* (see cy_basic_types.pyx)
        :param datetime: optional datetime object or list of datetime objects for which the value is desired
        :type datetime: datetime object
        :returns: numpy array containing dtype=basic_types.datetime_value_2d. Contains user specified datetime
        and the corresponding values in user specified data_format
        """
        if datetime is None:
            datetimeval = convert.to_datetime_value_2d(self.ossm.timeseries, data_format)
        else:
            datetime = np.asarray(datetime, dtype=np.datetime64).reshape(-1,)
            timeval = np.zeros((len(datetime),),dtype=basic_types.time_value_pair)
            timeval['time'] = time_utils.date_to_sec(datetime)
            timeval['value'] = self.ossm.get_time_value(timeval['time'])
            datetimeval = convert.to_datetime_value_2d(timeval, data_format)
            
        return datetimeval
    
    def set_timeseries(self, datetime_value_2d, data_format=basic_types.data_format.magnitude_direction):
        """
        sets the timeseries of the Wind object to the new value given by a numpy array. 
        The data_format for the input data defaults to 
        basic_types.data_format.magnitude_direction but can be changed by the user
        
        :param datetime_value_2d: timeseries of wind data defined in a numpy array
        :type datetime_value_2d: numpy array of dtype basic_types.datetime_value_2d
        :param data_format: output format for the times series; as defined by basic_types.data_format.
        :type data_format: integer value defined by basic_types.data_format.* (see cy_basic_types.pyx)
        """
        timeval = convert.to_time_value_pair(datetime_value_2d, data_format)
        self.ossm.timeseries = timeval
    
    
        