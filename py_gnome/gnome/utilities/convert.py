"""
Helper classes to do common tasks in pyGnome like;
- convert_formats to convert datetime_value to time_value_pair 
"""

from gnome.utilities import time_utils, transforms
from gnome import basic_types
import numpy as np

def to_time_value_pair(datetime_value_2d, in_data_format):
    """
    converts a numpy array containing basic_types.datetime_value_2d in user specified basic_types.data_format
    into a time_value_pair array 
    
    :param datetime_value_2d: numpy array of type basic_types.datetime_value_2d
    :param in_data_format: format of the array defined by one of the options given in basic_types.data_format
    """
    if datetime_value_2d.dtype is not basic_types.datetime_value_2d:
        raise ValueError("Method expects a numpy array containing basic_types.datetime_value_2d")
    
    # convert datetime_value_2d to time_value_pair
    time_value_pair = np.zeros((len(datetime_value_2d),), dtype=basic_types.time_value_pair)
    
    if in_data_format == basic_types.data_format.magnitude_direction:
        time_value_pair['time'] = time_utils.date_to_sec(datetime_value_2d['time'])
        uv = transforms.r_theta_to_uv_wind(datetime_value_2d['value'])
        time_value_pair['value']['u'] = uv[:,0]
        time_value_pair['value']['v'] = uv[:,1]
        
    elif in_data_format == basic_types.data_format.wind_uv:
        time_value_pair['time'] = time_utils.date_to_sec(datetime_value_2d['time'])
        time_value_pair['value']['u'] = datetime_value_2d['value'][:,0]
        time_value_pair['value']['v'] = datetime_value_2d['value'][:,1]
        
    else:
        raise ValueError("in_data_format is not one of the two supported types: basic_types.data_format.magnitude_direction, basic_types.data_format.wind_uv")
    
    return time_value_pair


def to_datetime_value_2d(time_value_pair, out_data_format):
    """
    converts a numpy array containing basic_types.time_value_pair to a numpy array containing basic_types.datetime_value_2d
    in user specified basic_types.data_format 
    
    :param time_value_pair: numpy array of type basic_types.time_value_pair
    :param out_data_format: desired format of the array defined by one of the options given in basic_types.data_format
    """
    if time_value_pair.dtype is not basic_types.time_value_pair:
        raise ValueError("Method expects a numpy array containing basic_types.time_value_pair")
    
    datetime_value_2d = np.zeros((len(time_value_pair),), dtype=basic_types.datetime_value_2d)
    
    # convert time_value_pair to datetime_value_2d in desired output format
    if out_data_format == basic_types.data_format.magnitude_direction:
        datetime_value_2d['time'] = time_utils.sec_to_date(time_value_pair['time'])

        uv = np.zeros((len(time_value_pair),2), dtype=np.double)
        uv[:,0] = time_value_pair['value']['u']
        uv[:,1] = time_value_pair['value']['v']
    
        datetime_value_2d['value'] = transforms.uv_to_r_theta_wind(uv)
        
    elif out_data_format == basic_types.data_format.wind_uv:
        datetime_value_2d['time'] = time_utils.sec_to_date(time_value_pair['time'])
        datetime_value_2d['value'][:,0]= time_value_pair['value']['u']
        datetime_value_2d['value'][:,1]= time_value_pair['value']['v']
    
    else:
        raise ValueError("out_data_format is not one of the two supported types: basic_types.data_format.magnitude_direction, basic_types.data_format.wind_uv")
    
    
    return datetime_value_2d