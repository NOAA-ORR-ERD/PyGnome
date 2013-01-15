import pytest

"""
Defines test fixtures

The scope="module" on the fixtures ensures it is only invoked once per test module
"""
import numpy as np
from datetime import datetime
from gnome import basic_types

"""
Skip slow tests
"""
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
        help="run slow tests")

def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")


"""
====================================
Following fixtures define standard functions for generating 
(r,theta) values and corresponding (u, v) values for testing
"""
@pytest.fixture(scope="module")
def invalid_rq():
    """
    Current transforms for wind and current from (r,theta) to (u,v)
    require r > 0, and 0 <= theta <=360
    
    :returns: MagDirectionUV object with bad_rq values and None for uv values
    """
    bad_rq = np.array( [(0,0),(-1,0),(1,-1),(1,361)], dtype=np.float64)
    return {'rq': bad_rq}

# use this for wind and current deterministic (r,theta)
rq = np.array( [(1, 0),(1,45),(1,90),(1,120),(1,180),(1,270)], dtype=np.float64)

@pytest.fixture(scope="module")
def rq_wind():
    """
    (r,theta) setup for wind on a unit circle for 0,90,180,270 deg
    """
    uv = np.array( [(0,-1),(-1./np.sqrt(2),-1./np.sqrt(2)),(-1,0),(-np.sqrt(3)/2,0.5),(0,1),(1,0)], dtype=np.float64)
    return {'rq': rq,'uv':uv}

@pytest.fixture(scope="module")
def rq_curr():
    """
    (r,theta) setup for current on a unit circle
    """
    uv = np.array( [(0,1),(1./np.sqrt(2),1./np.sqrt(2)),(1,0),(np.sqrt(3)/2,-0.5),(0,-1),(-1,0)], dtype=np.float64)
    return {'rq': rq,'uv':uv}
    

@pytest.fixture(scope="module")
def rq_rand():
    """
    (r,theta) setup randomly generated array of length = 3. The uv = None, only (r,theta)
    are randomly generated: 'r' is between (0,3) and 'theta' is between (0,360)
    """
    rq = np.zeros((5,2), dtype=np.float64)
    
    # cannot be 0 magnitude vector - let's just make it from 0.5
    rq[:,0] = np.random.uniform(.5,len(rq),len(rq))
        
    rq[:,1] = np.random.uniform(0,360,len(rq))
    return {'rq': rq}

"""
End fixtures for standard (r, theta) generation for Wind
====================================
====================================
Following fixtures define objects for testing the model and movers individually:
- Wind object
"""
@pytest.fixture(scope="module")
def wind_circ(rq_wind):
    """
    Create Wind object using the time series given by the test fixture 'rq_wind' 
    
    """
    from gnome import weather
    dtv_rq = np.zeros((len(rq_wind['rq']),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(2012,11,06,20,10+i,30) for i in range(len(dtv_rq))]
    dtv_rq.value = rq_wind['rq']
    dtv_uv = np.zeros((len(dtv_rq),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.time = dtv_rq.time
    dtv_uv.value= rq_wind['uv']
    wm  = weather.Wind(timeseries=dtv_rq,data_format=basic_types.data_format.magnitude_direction,units='meters per second')
    return {'wind':wm, 'rq': dtv_rq, 'uv': dtv_uv}

"""
End fixtures for testing model
====================================
"""
