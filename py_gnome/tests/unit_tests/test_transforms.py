from gnome.utilities.transforms import *
import numpy as np

import pytest

def test_exceptions():
    with pytest.raises(ValueError):
        r_theta_to_xy(np.array([0, 0]))
        r_theta_to_xy(np.array([-1, 0]))
        r_theta_to_xy(np.array([1, -1]))
        r_theta_to_xy(np.array([1, 361]))
    
rq      = np.zeros((4,2), dtype=np.double)
rq[:,0] = 1
rq[:,1] = [0, 90, 180, 270]
uv = np.array([[0,1],[-1,0],[0,-1],[1,0]])
        
def test_r_theta_to_uv_wind():    
    uv_out = r_theta_to_uv_wind(rq)
    print uv
    print uv_out
    assert np.all( uv == uv_out )
    
def test_wind_uv_to_r_theta():
    rq_out = wind_uv_to_r_theta(uv)
    print rq_out
    print rq
    assert np.all( rq == rq_out )
    
def test_r_theta_to_xy():
    x=np.zeros((4,2), dtype=np.double)
    x[:,0]=[i for i in range(1,9,2)]
    x[:,1]=[i for i in range(0,270,70)]
    xy = xy_to_r_theta( r_theta_to_xy(x))
    print xy
    print x
    #assert np.all( x[:,0] == rq[:,0])    # not sure why this fails?
    assert np.allclose(xy, x, 1e-14, 1e-14)
    
    
def test_xy_to_r_theta():
    x=np.zeros((4,2), dtype=np.double)
    x[:,0]=[1,0,-1,0]
    x[:,1]=[0,1,0,-1]
    uv = r_theta_to_xy(xy_to_r_theta(x))
    assert np.allclose(uv, x, 1e-14, 1e-14)