"""
Helper classes to do common tasks in pyGnome like;
- convert_formats to convert datetime_value to time_value_pair
"""

from gnome.utilities import time_utils, transforms
from gnome import basic_types
import numpy as np


def to_time_value_pair(datetime_value, in_ts_format=None):
    """
    converts a numpy array containing basic_types.datetime_value_2d in
    user specified basic_types.ts_format into a time_value_pair array
    or it takes a basic_types.datetime_value_1d array and converts it to
    a time_value_pair array -- for 1d data, assume the ['value'] contains
    the 'u' component and set the 'v' component to 0.0

    :param datetime_value: numpy array of type basic_types.datetime_value_2d or
        basic_types.datetime_value_1d
    :param in_ts_format=None: format of the datetime_value_2d array - not
        required when converting from datetime_value_1d. Can be defined by a
        string 'r-theta', 'uv' or by an integer defined by one of the options
        given in basic_types.ts_format.
    """

    if(datetime_value.dtype != basic_types.datetime_value_2d and
        datetime_value.dtype != basic_types.datetime_value_1d):
        raise ValueError('Method expects a numpy array containing '
            'basic_types.datetime_value_2d or basic_types.datetime_value_1d')

    # convert datetime_value_2d to time_value_pair

    time_value_pair = np.zeros((len(datetime_value), ),
                               dtype=basic_types.time_value_pair)

    time_value_pair['time'] = \
            time_utils.date_to_sec(datetime_value['time'])
    if datetime_value.dtype == basic_types.datetime_value_1d:
        time_value_pair['value']['u'] = datetime_value['value'][:, 0]

    else:
        if in_ts_format is None:
            raise ValueError("for datetime_value_2d data conversion, the "
                "format defined by 'in_ts_format', cannot be None ")
        if isinstance(in_ts_format, basestring):
            in_ts_format = tsformat(in_ts_format)

        if in_ts_format == basic_types.ts_format.magnitude_direction:
            uv = transforms.r_theta_to_uv_wind(datetime_value['value'])
            time_value_pair['value']['u'] = uv[:, 0]
            time_value_pair['value']['v'] = uv[:, 1]

        elif in_ts_format == basic_types.ts_format.uv:
            time_value_pair['value']['u'] = datetime_value['value'][:, 0]
            time_value_pair['value']['v'] = datetime_value['value'][:, 1]
        else:

            raise ValueError('in_ts_format is not one of the two supported '
                'types: basic_types.ts_format.magnitude_direction, '
                'basic_types.ts_format.uv')

    return time_value_pair


def to_datetime_value_2d(time_value_pair, out_ts_format):
    """
    converts a numpy array containing basic_types.time_value_pair to a
    numpy array containing basic_types.datetime_value_2d in user specified
    basic_types.ts_format

    :param time_value_pair: numpy array of type basic_types.time_value_pair
    :param out_ts_format: desired format of the array defined by one of the
                          options given in basic_types.ts_format
    """

    if time_value_pair.dtype != basic_types.time_value_pair:
        raise ValueError('Method expects a numpy array containing basic_types.time_value_pair'
                         )

    datetime_value_2d = np.zeros((len(time_value_pair), ),
                                 dtype=basic_types.datetime_value_2d)

    if isinstance(out_ts_format, basestring):
        out_ts_format = tsformat(out_ts_format)

    # convert time_value_pair to datetime_value_2d in desired output format

    if out_ts_format == basic_types.ts_format.magnitude_direction:
        datetime_value_2d['time'] = \
            time_utils.sec_to_date(time_value_pair['time'])

        uv = np.zeros((len(time_value_pair), 2), dtype=np.double)
        uv[:, 0] = time_value_pair['value']['u']
        uv[:, 1] = time_value_pair['value']['v']

        datetime_value_2d['value'] = transforms.uv_to_r_theta_wind(uv)
    elif out_ts_format == basic_types.ts_format.uv:

        datetime_value_2d['time'] = \
            time_utils.sec_to_date(time_value_pair['time'])
        datetime_value_2d['value'][:, 0] = time_value_pair['value']['u']
        datetime_value_2d['value'][:, 1] = time_value_pair['value']['v']
    else:

        raise ValueError('out_ts_format is not one of the two supported types: basic_types.ts_format.magnitude_direction, basic_types.ts_format.uv'
                         )

    return datetime_value_2d


def to_datetime_value_1d(time_value_pair):
    '''
    converts a numpy array containing basic_types.time_value_pair to a
    numpy array containing basic_types.datetime_value_1d. It simply drops the
    2nd component of value: eg:

      time_value_pair[0] = ((datetime), (val_0, val_1))

    becomes: output = ((datetime), (val_0,))
    This is partly used for timeseries for Tide data and in this case, the
    v-component of velocity (u, v) is 0.0

    This function does not perform a check to see if 2nd component is 0.0, it
    simply drops it.

    :param time_value_pair: numpy array of type basic_types.time_value_pair
    '''
    if time_value_pair.dtype != basic_types.time_value_pair:
        raise ValueError('Method expects a numpy array containing '
            'basic_types.time_value_pair')

    datetime_value_1d = np.zeros((len(time_value_pair), ),
                                 dtype=basic_types.datetime_value_1d)
    datetime_value_1d['time'] = \
            time_utils.sec_to_date(time_value_pair['time'])
    datetime_value_1d['value'][:, 0] = time_value_pair['value']['u']

    return datetime_value_1d


def tsformat(format):
    """
    convert string 'uv' or 'magnitude_direction' into appropriate integer given by basic_types.ts_format.*
    """

    if format == 'r-theta':
        return basic_types.ts_format.magnitude_direction
    elif format == 'uv':
        return basic_types.ts_format.uv
    else:
        raise ValueError("timeseries format can only be 'r-theta' or 'uv', the format entered is not recognized as valid format"
                         )


