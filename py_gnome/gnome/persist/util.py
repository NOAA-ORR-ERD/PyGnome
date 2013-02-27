'''
Created on Feb 26, 2013

functions for converting PyGnome data-structs (numpy arrays, datetime) to plain python structs
these can be used by schema's before validation and also by json_serialize
'''

import dateutil.parser
import numpy

import gnome

def str_to_date(obj):
    """
    If string is not valid datetime iso format string, this will throw ValueError
    """
    if isinstance(obj, str):
        return dateutil.parser.parse(obj)
    
def datetime_value_2d_to_list(np_timeseries):
    series = []
        
    for wind_value in np_timeseries:
        dt = wind_value[0].astype(object).isoformat()
        series.append((dt, wind_value[1][0], wind_value[1][1]))
    return series
    
def list_to_datetime_value_2d(series):
    num_series = len(series)
    timeseries = numpy.zeros((num_series,),
                          dtype=gnome.basic_types.datetime_value_2d)
    
    for idx, value in enumerate(series):
        timeseries['time'][idx] = value[0]
        timeseries['value'][idx] = (value[1], value[2])

    return timeseries
    