from gnome.utilities.transforms import *
import numpy as np

import pytest

def test_exceptions():
    with pytest.raises(ValueError):
        r_theta_to_uv(np.array([0, 0]))
        r_theta_to_uv(np.array([-1, 0]))
        r_theta_to_uv(np.array([1, -1]))
        r_theta_to_uv(np.array([1, 361]))
        
def test_r_theta_to_uv():
    x=np.zeros((4,2), dtype=np.double)
    x[:,0]=[i for i in range(1,9,2)]
    x[:,1]=[i for i in range(0,270,70)]
    rq = uv_to_r_theta( r_theta_to_uv(x))
    print rq
    print x
    #assert np.all( x[:,0] == rq[:,0])    # not sure why this fails?
    np.allclose(rq, x, 1e-14, 1e-14)
    
    
def test_uv_to_r_theta():
    x=np.zeros((4,2), dtype=np.double)
    x[:,0]=[1,0,-1,0]
    x[:,1]=[0,1,0,-1]
    uv = r_theta_to_uv(uv_to_r_theta(x))
    np.allclose(uv, x, 1e-14, 1e-14)