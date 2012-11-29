"""
Unit tests ConvertDatetimeValue methods 
"""

from gnome import basic_types
from gnome.helpers import convert
import pytest
import numpy as np

# set up test cases
dtv_rq = np.zeros((3,), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
dtv_rq.value = [(1,0),
                (np.sqrt(2),45),
                (1,120)]

dtv_uv = np.zeros((len(dtv_rq),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
dtv_uv.value = [(0,-1),
                (-1,-1),
                (-np.sqrt(3)/2,0.5)]

tv = np.zeros((len(dtv_uv),), dtype=basic_types.time_value_pair).view(dtype=np.recarray)
tv.time = 8*3600.    # zero for datetime numpy array
tv.value.u = dtv_uv.value[:,0]
tv.value.v = dtv_uv.value[:,1]

def test_exceptions():
    """
    test exceptions
    """
    with pytest.raises(ValueError):
        convert.to_time_value_pair(tv, basic_types.data_format.magnitude_direction)
        convert.to_time_value_pair(dtv, -1)
        convert.to_datetime_value_2d(dtv, basic_types.data_format.magnitude_direction)
        convert.to_datetime_value_2d(tv, -1)
        convert.to_datetime_value_2d(tv, basic_types.data_format.magnitude_direction)
        
def test_to_time_value_pair():
    out_tv = convert.to_time_value_pair(dtv_rq, in_data_format=basic_types.data_format.magnitude_direction)
    print out_tv.value.u
    print tv.value.u
    assert np.all( tv.time == tv.time)
    assert np.allclose(tv.value.u, out_tv.value.u, 1e-10, 1e-10)
    assert np.allclose(tv.value.v, out_tv.value.v, 1e-10, 1e-10)
    
def test_to_datetime_value_2d_rq():
    out_dtval = convert.to_datetime_value_2d(tv, out_data_format=basic_types.data_format.magnitude_direction)
    print "-----"
    print out_dtval.time
    print dtv_rq.time
    print out_dtval.value
    print dtv_rq.value
    print "-----"
    assert np.all( out_dtval.time == dtv_rq.time)
    assert np.allclose( out_dtval.value, dtv_rq.value, 1e-10, 1e-10) 

def test_to_datetime_value_2d_uv():
    out_dtval = convert.to_datetime_value_2d(tv, out_data_format=basic_types.data_format.wind_uv)
    print "-----"
    print out_dtval.time
    print dtv_uv.time
    print out_dtval.value
    print dtv_uv.value
    print "-----"
    assert np.all( out_dtval.time == dtv_rq.time)
    assert np.allclose( out_dtval.value, dtv_uv.value, 1e-10, 1e-10)

if __name__=="__main__":
    test_to_time_value_pair()
    test_to_datetime_value_2d_rq()
    test_to_datetime_value_2d_uv()