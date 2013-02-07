"""
Unit tests ConvertDatetimeValue methods 
"""

from gnome import basic_types
from gnome.utilities import convert
import pytest
import numpy as np

"""
Define a wind_ts (wind timeseries) fixture for defining the truth (datetime, r, theta), 
and the corresponding (datetime, u,v) and time_value_pair arrays.

If this fixture is used by a different module, then move it to conftest.py
Currently, it is only used by test_helpers_convert.py
"""
@pytest.fixture(scope="module")
def wind_ts(rq_wind):
    """
    setup a wind timeseries - uses the rq_wind fixture to get the wind values used for conversions
    - returns a dict with the expected datetime_rq, datetime_uv and time_value_pair objects
    """
    dtv_rq = np.zeros((len(rq_wind['rq']),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.value = rq_wind['rq']

    dtv_uv = np.zeros((len(dtv_rq),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.value = rq_wind['uv']

    tv = np.zeros((len(dtv_uv),), dtype=basic_types.time_value_pair).view(dtype=np.recarray)

    # for np.zeros for datetime, this is the time in sec. It is 8hours offset from GMT - need to make this locale independent
    tv.time = 8*3600
    tv.value.u = rq_wind['uv'][:,0]
    tv.value.v = rq_wind['uv'][:,1]

    print "Test Case - actual values:"
    print "datetime_value_2d: datetime, (r, theta):"
    print dtv_rq.time
    print dtv_rq.value
    print "----------"
    print "datetime_value_2d: datetime, (u, v):"
    print dtv_uv.time
    print dtv_uv.value
    print "----------"
    print "time_value_pair: time, (u, v):"
    print tv.time
    print tv.value.reshape(len(tv.value),-1)

    return {'dtv_rq':dtv_rq,'dtv_uv':dtv_uv, 'tv':tv}


def test_exceptions(wind_ts,invalid_rq):
    """
    test exceptions
    """
    with pytest.raises(ValueError):
        convert.to_time_value_pair(wind_ts['tv'], basic_types.ts_format.magnitude_direction)
        convert.to_time_value_pair(wind_ts['dtv'], -1)
        convert.to_datetime_value_2d(wind_ts['tv'], -1)
        convert.to_datetime_value_2d(wind_ts['tv'], basic_types.ts_format.magnitude_direction)

    # following also raises ValueError. This gives invalid (r,theta) inputs which are rejected
    # by the transforms.r_theta_to_uv_wind method. It tests the inner exception is correct
    with pytest.raises(ValueError):
        invalid_dtv_rq = np.zeros((len(invalid_rq['rq']),), dtype=basic_types.datetime_value_2d)
        invalid_dtv_rq['value'] = invalid_rq['rq']
        convert.to_time_value_pair( invalid_dtv_rq, basic_types.ts_format.magnitude_direction)

# tolerance for numpy.allclose(...)
atol = 1e-14
rtol = 0
        
def test_to_time_value_pair(wind_ts):
    out_tv = convert.to_time_value_pair(wind_ts['dtv_rq'], in_ts_format=basic_types.ts_format.magnitude_direction).view(dtype=np.recarray)
    assert np.all( wind_ts['tv'].time == out_tv.time)
    assert np.allclose( wind_ts['tv'].value.u, out_tv.value.u, atol, rtol)
    assert np.allclose( wind_ts['tv'].value.v, out_tv.value.v, atol, rtol)
    
def test_to_datetime_value_2d_rq(wind_ts):
    out_dtval = convert.to_datetime_value_2d(wind_ts['tv'], out_ts_format=basic_types.ts_format.magnitude_direction).view(dtype=np.recarray)
    assert np.all( out_dtval.time == wind_ts['dtv_rq'].time)
    assert np.allclose( out_dtval.value, wind_ts['dtv_rq'].value, atol, rtol) 

def test_to_datetime_value_2d_uv(wind_ts):
    out_dtval = convert.to_datetime_value_2d(wind_ts['tv'], out_ts_format=basic_types.ts_format.uv).view(dtype=np.recarray)
    assert np.all( out_dtval.time == wind_ts['dtv_rq'].time)
    assert np.allclose( out_dtval.value, wind_ts['dtv_uv'].value, atol, rtol)
