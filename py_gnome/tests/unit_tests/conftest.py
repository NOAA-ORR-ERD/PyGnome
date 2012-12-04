import pytest

"""
Defines test fixtures

The scope="module" on the fixtures ensures it is only invoked once per test module
"""
import numpy as np

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
def wind_circ():
    """
    (r,theta) setup for wind on a unit circle for 0,90,180,270 deg
    """
    uv = np.array( [(0,-1),(-1./np.sqrt(2),-1./np.sqrt(2)),(-1,0),(-np.sqrt(3)/2,0.5),(0,1),(1,0)], dtype=np.float64)
    return {'rq': rq,'uv':uv}

@pytest.fixture(scope="module")
def curr_circ():
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
    rq = np.zeros((3,2), dtype=np.float64)
    rq[:,0] = np.random.uniform(0,3,len(rq))
    rq[:,1] = np.random.uniform(0,360,len(rq))
    return {'rq': rq}
