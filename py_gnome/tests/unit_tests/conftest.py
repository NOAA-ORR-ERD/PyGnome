"""
Defines test fixtures

The scope="module" on the fixtures ensures it is only invoked once per test module
"""
import sys, os
from datetime import datetime

import numpy as np
import pytest

from gnome import basic_types
from gnome.utilities import rand

def pytest_addoption(parser):
    '''
    Skip slow tests
    '''
    parser.addoption("--runslow", action="store_true",
        help="run slow tests")

def pytest_runtest_setup(item):
    """
    pytest builtin hook
    
    This is executed before pytest_runtest_call. 
    pytest_runtest_call is invoked to execute the test item. So the code in here
    is executed before each test.
    """
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
        
    # set random seed:
    print "Seed C++, python, numpy random number generator to 1"
    rand.seed(1)

def pytest_sessionstart():
    from py.test import config

    # Only run database setup on master (in case of xdist/multiproc mode)
    if not hasattr(config, 'slaveinput'):
        try:
            from gnome.db.oil_library.initializedb import initialize_sql, load_database

            data_dir = get_data_dir()
            oillib_file = os.path.join(data_dir, r'OilLib.smaller')
            db_file = os.path.join(data_dir, r'OilLibrary.db')
            sqlalchemy_url = 'sqlite:///{0}'.format(db_file)
            settings = {'sqlalchemy.url': sqlalchemy_url,
                        'oillib.file': oillib_file
                        }
            initialize_sql(settings)
            load_database(settings)
        except ImportError as ie:
            print "\nWarning: Required modules for database unit-testing not found."
            dependant_modules = ('sqlalchemy','zope.sqlalchemy','transaction')
            print ie
            print "Also may need:",
            print '\t {0}\n'.format([m for m in dependant_modules if not m in sys.modules])

def get_data_dir():
    data_dir = os.path.dirname(__file__)
    return os.path.join(data_dir, r'SampleData/oil_library')

"""
====================================
Following fixtures define standard functions for generating 
(r,theta) values and corresponding (u, v) values for testing
"""
@pytest.fixture(scope="module")
def invalid_rq():
    """
    Provides invalid (r,theta) values for the transforms for wind 
    and current from (r,theta) to (u,v)
    
    Transforms require r > 0, and 0 <= theta <=360. This returns bad values
    for (r,q) 
    
    :returns: dictionary containing 'rq' which is numpy array of '(r,q)' values
    that violate above requirement
    """
    bad_rq = np.array( [(-1,0),(1,-1),(1,361)], dtype=np.float64)
    return {'rq': bad_rq}

# use this for wind and current deterministic (r,theta)
rq = np.array( [(1, 0),(1,45),(1,90),(1,120),(1,180),(1,270)], dtype=np.float64)

@pytest.fixture(scope="module")
def rq_wind():
    """
    (r,theta) setup for wind on a unit circle for 0,90,180,270 deg
    
    :returns: dictionary containing 'rq' and 'uv' which is numpy array of (r,q) values and the 
    corresponding (u,v)
    """
    uv = np.array( [(0,-1),(-1./np.sqrt(2),-1./np.sqrt(2)),(-1,0),(-np.sqrt(3)/2,0.5),(0,1),(1,0)], dtype=np.float64)
    return {'rq': rq,'uv':uv}

@pytest.fixture(scope="module")
def rq_curr():
    """
    (r,theta) setup for current on a unit circle
    
    :returns: dictionary containing 'rq' and 'uv' which is numpy array of (r,q) values and the 
    corresponding (u,v)
    """
    uv = np.array( [(0,1),(1./np.sqrt(2),1./np.sqrt(2)),(1,0),(np.sqrt(3)/2,-0.5),(0,-1),(-1,0)], dtype=np.float64)
    return {'rq': rq,'uv':uv}
    

@pytest.fixture(scope="module")
def rq_rand():
    """
    (r,theta) setup randomly generated array of length = 3. The uv = None, only (r,theta)
    are randomly generated: 'r' is between (.5,len(rq)) and 'theta' is between (0,360)
    
    :returns: dictionary containing randomly generated 'rq', which is numpy array of (r,q) values
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
    'wind' object where timeseries is defined as:
         - 'time' defined by: [datetime(2012,11,06,20,10+i,30) for i in range(len(dtv_rq))]
         - 'value' defined by: (r,theta) values ferom rq_wind fixtures, units are 'm/s'
    
    :returns: a dict containing following three keys: 'wind', 'rq', 'uv'
              'wind' object, timeseries in (r,theta) format 'rq', timeseries in (u,v) format 'uv'. 
    """
    from gnome import environment
    dtv_rq = np.zeros((len(rq_wind['rq']),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(2012,11,06,20,10+i,30) for i in range(len(dtv_rq))]
    dtv_rq.value = rq_wind['rq']
    dtv_uv = np.zeros((len(dtv_rq),), dtype=basic_types.datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.time = dtv_rq.time
    dtv_uv.value= rq_wind['uv']
    wm  = environment.Wind(timeseries=dtv_rq,format='r-theta',units='meter per second')
    return {'wind':wm, 'rq': dtv_rq, 'uv': dtv_uv}

@pytest.fixture(scope="module")
def sample_model_spatial_release_spill():
    """ 
    sample model with no outputter
    Uses:
        SampleData/MapBounds_Island.bna
        SpatialReleaseSpill with 10 particles
        RandomMover
        duration is 1 hour with 15min intervals so 5 timesteps total, including initial condition
        model is uncertain and cache is not enabled
    """
    import gnome
    from datetime import datetime,timedelta
    
    start_time = datetime(2012, 9, 15, 12, 0)
    
    # the image output map
    mapfile = os.path.join(os.path.dirname(__file__),'SampleData','MapBounds_Island.bna')

    # the land-water map
    map = gnome.map.MapFromBNA( mapfile,
                                refloat_halflife=6*3600, #seconds
                                )

    model = gnome.model.Model(time_step=timedelta(minutes=15), 
                              start_time=start_time,
                              duration=timedelta(hours=1),
                              map=map,
                              uncertain=True,
                              cache_enabled=False,)

    model.movers += gnome.movers.RandomMover(diffusion_coef=100000)

    N = 10 # a line of ten points
    start_points = np.zeros((N, 3) , dtype=np.float64)
    start_points[:,0] = np.linspace(-127.1, -126.5, N)
    start_points[:,1] = np.linspace( 47.93, 48.1, N)
    
    spill = gnome.spill.SpatialReleaseSpill(start_positions = start_points,
                                            release_time = start_time,
                                            )

    model.spills += spill
    model.start_time = spill.release_time

    model.uncertain = True
    return {'model':model}

"""
End fixtures for testing model
====================================
"""
