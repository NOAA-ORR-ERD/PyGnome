from gnome.utilities.transforms import *
import numpy as np
import random
import pytest

def test_exceptions():
    with pytest.raises(ValueError):
        r_theta_to_uv_wind(np.array([0, 0]))
        r_theta_to_uv_wind(np.array([-1, 0]))
        r_theta_to_uv_wind(np.array([1, -1]))
        r_theta_to_uv_wind(np.array([1, 361]))
    

rq = np.array( [(1, 0),(1,90),(1,180),(1,270)], dtype=np.float64)
uv = np.array( [(0,-1),(-1,0),(0,  1),(1,  0)], dtype=np.float64)
    
def test_r_theta_to_uv_wind():
    uv_out = r_theta_to_uv_wind(rq)
    print uv_out
    print uv
    assert np.all( uv_out == uv)
    
    
def test_uv_to_r_theta_wind():
    rq_out = uv_to_r_theta_wind(uv)
    print rq_out
    print rq
    assert np.all( rq_out == rq)
    
def test_wind_inverse():
    """
    randomly generates an (r,theta) and applies the transform to convert to (u,v), then back to (r,theta).
    It checks the result is accurate to within 10-10 tolerance
    """
    rq = np.array([(random.uniform(0,1), random.uniform(0,360))], dtype=np.float64)
    rq_out = uv_to_r_theta_wind( r_theta_to_uv_wind(rq))
    print rq
    print rq_out
    assert np.allclose(rq, rq_out, 1e-10, 1e-10)
    
rq_c = np.array( [(1, 0),(np.sqrt(2),45),(1,90),(1,180),(1,270)], dtype=np.float64)
uv_c = np.array( [(0, 1),         (1, 1),(1, 0),(0, -1),(-1, 0)], dtype=np.float64)
def test_r_theta_to_uv_current():
    uv_out = r_theta_to_uv_current(rq_c)
    assert np.all( uv_out == uv_c )
    
def test_uv_to_r_theta_current():
    rq_out = uv_to_r_theta_current(uv_c)
    assert np.all( rq_out == rq_c )
    
def test_current_inverse():
    """
    randomly generates an (r,theta) and applies the transform to convert to (u,v), then back to (r,theta).
    It checks the result is accurate to within 10-10 tolerance
    """
    rq = np.array([(random.uniform(0,1), random.uniform(0,360))], dtype=np.float64)
    rq_out = uv_to_r_theta_current( r_theta_to_uv_current(rq))
    print rq
    print rq_out
    assert np.allclose(rq, rq_out, 1e-10, 1e-10)
