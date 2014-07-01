"""
Defines test fixtures as well as functions that are used by multiple tests
and test modules

The scope="module" on the fixtures ensures it is only invoked once per test
module
"""

import os
from datetime import datetime, timedelta
import copy
import shutil

import numpy
np = numpy

import pytest

import gnome
from gnome.basic_types import datetime_value_2d
from gnome.array_types import windages, windage_range, windage_persist

from gnome.map import MapFromBNA
from gnome.model import Model

from gnome.spill_container import SpillContainer

from gnome.movers import SimpleMover
from gnome.persist import load
from gnome.utilities.remote_data import get_datafile


base_dir = os.path.dirname(__file__)


def mock_append_data_arrays(array_types, num_elements, data_arrays={}):
    """
    takes array_types desired by test function and number of elements
    to be initialized. For testing element_type functionality with a
    SpillContainer. Mocks the functionality of
    SpillContainer()._append_data_arrays(..)

    :param array_types: dict of array_types used to initialize data_arrays.
        If data_arrays is empty, then use the initialize_null() method of
        each array_type to first initialize it, then append to it.
    :param num_elements: number of elements to be released. Append
        'num_elements' to data_arrays
    :param data_arrays: empty by default. But this could contain a dictionary
        of data_arrays. In this case, just initialize each array_type for
        'num_elements' and append it to numpy array in dict.
        Function makes a deepcopy of data_arrays, appends to the copied dict,
        and returns this copy after appending num_elements to numpy arrays.
        SpillContainer would be managing this dict in the real use case
    """
    array_types = dict(array_types)
    data_arrays = copy.deepcopy(data_arrays)

    for name, array_type in array_types.iteritems():
        # initialize null arrays so they exist before appending
        if name not in data_arrays:
            data_arrays[name] = array_type.initialize_null()

        arr = array_type.initialize(num_elements)
        data_arrays[name] = np.r_[data_arrays[name], arr]

    return data_arrays


def sample_sc_release(num_elements=10,
                      start_pos=(0.0, 0.0, 0.0),
                      release_time=datetime(2000, 1, 1, 1),
                      uncertain=False,
                      time_step=360,
                      spill=None,
                      element_type=None,
                      current_time=None,
                      arr_types=None):
    """
    Initialize a Spill of type 'spill', add it to a SpillContainer.
    Invoke release_elements on SpillContainer, then return the spill container
    object

    If 'spill' is None, define a Spill object with a PointLineRelease type
    of release
    """
    if current_time is None:
        current_time = release_time

    if spill is None:
        spill = gnome.spill.point_line_release_spill(num_elements, start_pos,
                                            release_time)
    spill.mass = num_elements

    if element_type is not None:
        spill.element_type = element_type

    if current_time is None:
        current_time = spill.release_time

    if arr_types is None:
        # default always has standard windage parameters required by wind_mover
        arr_types = {'windages': windages,
                     'windage_range': windage_range,
                     'windage_persist': windage_persist}

    sc = SpillContainer(uncertain)
    sc.spills.add(spill)

    # used for testing so just assume there is a Windage array
    sc.prepare_for_model_run(arr_types)
    sc.release_elements(time_step, current_time)
    return sc


def testdata():
    'define all the testdata files here'
    s_data = os.path.join(base_dir, 'sample_data')
    lis = os.path.join(s_data, 'long_island_sound')
    dbay = os.path.join(s_data, 'delaware_bay')
    curr_dir = os.path.join(s_data, 'currents')
    tide_dir = os.path.join(s_data, 'tides')
    wind_dir = os.path.join(s_data, 'winds')
    testmap = os.path.join(base_dir, '../sample_data', 'MapBounds_Island.bna')
    bna_sample = os.path.join(s_data, r"MapBounds_2Spillable2Islands2Lakes.bna")

    data = dict()

    data['CatsMover'] = \
        {'curr': get_datafile(os.path.join(lis, 'tidesWAC.CUR')),
         'tide': get_datafile(os.path.join(lis, 'CLISShio.txt'))}
    data['ComponentMover'] = \
        {'curr': get_datafile(os.path.join(dbay, 'NW30ktwinds.cur')),
         'wind': get_datafile(os.path.join(dbay, 'ConstantWind.WND'))}
    data['CurrentCycleMover'] = \
        {'curr': get_datafile(os.path.join(curr_dir, 'PQBayCur.nc4')),
         'topology': get_datafile(os.path.join(curr_dir, 'PassamaquoddyTOP.dat')),
         'tide': get_datafile(os.path.join(tide_dir, 'EstesHead.txt'))}
    data['GridCurrentMover'] = \
        {'curr': get_datafile(os.path.join(curr_dir, 'ChesBay.nc')),
         'topology': get_datafile(os.path.join(curr_dir, 'ChesBay.dat'))}
    data['GridWindMover'] = \
        {'wind': get_datafile(os.path.join(wind_dir, 'WindSpeedDirSubset.nc')),
         'topology': get_datafile(os.path.join(wind_dir, 'WindSpeedDirSubsetTop.dat'))}
    data['MapFromBNA'] = {'testmap': testmap}
    data['Renderer'] = {'bna_sample': bna_sample,
                        'output_dir': os.path.join(base_dir, 'renderer_output')}

    return data


@pytest.fixture(scope='module')
def invalid_rq():
    """
    Provides invalid (r,theta) values for the transforms for wind
    and current from (r,theta) to (u,v)

    Transforms require r > 0, and 0 <= theta <=360. This returns bad values
    for (r,q)

    :returns: dictionary containing 'rq' which is numpy array of '(r,q)' values
    that violate above requirement
    """

    bad_rq = np.array([(-1, 0), (1, -1), (1, 361)], dtype=np.float64)
    return {'rq': bad_rq}

# use this for wind and current deterministic (r,theta)

rq = np.array([
    (1, 0),
    (1, 45),
    (1, 90),
    (1, 120),
    (1, 180),
    (1, 270),
    ], dtype=np.float64)


@pytest.fixture(scope='module')
def rq_wind():
    """
    (r,theta) setup for wind on a unit circle for 0,90,180,270 deg

    :returns: dictionary containing 'rq' and 'uv' which is numpy array of (r,q)
        values and the corresponding (u,v)
    """

    uv = np.array([
        (0, -1),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1, 0),
        (-np.sqrt(3) / 2, .5),
        (0, 1),
        (1, 0),
        ], dtype=np.float64)
    return {'rq': rq, 'uv': uv}


@pytest.fixture(scope='module')
def rq_curr():
    """
    (r,theta) setup for current on a unit circle

    :returns: dictionary containing 'rq' and 'uv' which is numpy array of (r,q)
        values and the corresponding (u,v)
    """

    uv = np.array([
        (0, 1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1, 0),
        (np.sqrt(3) / 2, -.5),
        (0, -1),
        (-1, 0),
        ], dtype=np.float64)
    return {'rq': rq, 'uv': uv}


@pytest.fixture(scope='module')
def rq_rand():
    """
    (r,theta) setup randomly generated array of length = 3. The uv = None,
    only (r,theta) are randomly generated: 'r' is between (.5,len(rq)) and
    'theta' is between (0,360)

    :returns: dictionary containing randomly generated 'rq', which is numpy
        array of (r,q) values
    """

    rq = np.zeros((5, 2), dtype=np.float64)

    # cannot be 0 magnitude vector - let's just make it from 0.5

    rq[:, 0] = np.random.uniform(.5, len(rq), len(rq))

    rq[:, 1] = np.random.uniform(0, 360, len(rq))
    return {'rq': rq}


@pytest.fixture(scope='module')
def sample_graph():
    from gnome.utilities.weathering.graphs import Graph

    return Graph(points=((1, 2, 3),
                         (2, 3, 4),
                         (3, 4, 5),),
                 labels=('x', 'F1(x)', 'F2(x)'),
                 formats=('', 'r-o', 'g->'),
                 title='Custom line styles'
                 )


@pytest.fixture(scope='module')
def wind_circ(rq_wind):
    """
    Create Wind object using the time series given by test fixture 'rq_wind'
    'wind' object where timeseries is defined as:
         - 'time' defined by: [datetime(2012,11,06,20,10+i,0)
            for i in range(len(dtv_rq))]
         - 'value' defined by: (r,theta) values ferom rq_wind fixtures, units
            are 'm/s'

    :returns: a dict containing following three keys: 'wind', 'rq', 'uv'
              'wind' object, timeseries in (r,theta) format 'rq', timeseries in
              (u,v) format 'uv'.
    """

    from gnome import environment
    dtv_rq = np.zeros((len(rq_wind['rq']), ),
                      dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(
        2012,
        11,
        06,
        20,
        10 + i,
        0,
        ) for i in range(len(dtv_rq))]
    dtv_rq.value = rq_wind['rq']
    dtv_uv = np.zeros((len(dtv_rq), ),
                   dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.time = dtv_rq.time
    dtv_uv.value = rq_wind['uv']
    wm = environment.Wind(timeseries=dtv_rq, format='r-theta',
                          units='meter per second')
    return {'wind': wm, 'rq': dtv_rq, 'uv': dtv_uv}


@pytest.fixture(scope='module')
def sample_spatial_release_spill():
    """
    creates an example SpatialRelease object with
    start_positions: ((0., 0., 0.), (28.0, -75.0, 0.), (-15, 12, 4.0),
                   (80, -80, 100.0))
    release_time: datetime(2012, 1, 1, 1)
    :returns: a tuple containing (spill, start_positions). start_positions
        should be equal to spill.start_positions
    """
    from gnome.spill import SpatialRelease
    start_positions = ((0., 0., 0.),
                       (28.0, -75.0, 0.),
                       (-15, 12, 4.0),
                       (80, -80, 100.0))
    rel = SpatialRelease(datetime(2012, 1, 1, 1), start_positions)
    sp = gnome.spill.Spill(release=rel)
    return (sp, start_positions)


@pytest.fixture(scope='module')
def sample_vertical_plume_spill():
    '''
    creates an example VerticalPlumeSource object
    '''
    from gnome.spill import VerticalPlumeRelease, Spill
    from gnome.utilities.plume import get_plume_data

    release_time = datetime.now()
    vps = VerticalPlumeRelease(num_elements=200,
                              start_position=(28, -78, 0.),
                              release_time=release_time,
                              end_release_time=release_time + timedelta(hours=24),
                              plume_data=get_plume_data(),
                              )

    vps.plume_gen.time_step_delta = timedelta(hours=1).total_seconds()
    return Spill(vps)


@pytest.fixture(scope='module')
def sample_sc_no_uncertainty():
    """
    Sample spill container with 2 point_line_release_spill spills:

    - release_time for 2nd spill is 1 hour delayed
    - 2nd spill takes 4 hours to release and end_position is different so it
      is a time varying, line release
    - both have a volume of 10 and default element_type

    Nothing is released. This module simply defines a SpillContainer, adds
    the two spills and returns it. It is used in test_spill_container.py
    and test_elements.py so defined as a fixture.
    """
    sc = SpillContainer()
    # Sample data for creating spill
    num_elements = 100
    start_position = (23.0, -78.5, 0.0)
    release_time = datetime(2012, 1, 1, 12)
    end_position = (24.0, -79.5, 1.0)
    end_release_time = datetime(2012, 1, 1, 12) + timedelta(hours=4)

    spills = [gnome.spill.point_line_release_spill(num_elements,
                              start_position, release_time, volume=10),
              gnome.spill.point_line_release_spill(num_elements,
                              start_position,
                              release_time + timedelta(hours=1),
                              end_position, end_release_time,
                              volume=10),
              ]
    sc.spills.add(spills)
    return sc


@pytest.fixture(scope='module')
def sample_model():
    """
    sample model with no outputter and no spills. Use this as a template for
    fixtures to add spills
    Uses:
        sample_data/MapBounds_Island.bna
        Contains: gnome.movers.SimpleMover(velocity=(1.0, -1.0, 0.0))
        duration is 1 hour with 15min intervals so 5 timesteps total,
        including initial condition,
        model is uncertain and cache is not enabled
        No spills or outputters defined

    To use:
        add a spill and run

    :returns: It returns a dict -
        {'model':model,
         'release_start_pos':start_points,
         'release_end_pos':end_points}
        The release_start_pos and release_end_pos can be used by test to define
        the spill's 'start_position' and 'end_position'
    """

    release_time = datetime(2012, 9, 15, 12, 0)

    # the image output map

    mapfile = os.path.join(os.path.dirname(__file__), '../sample_data',
                           'MapBounds_Island.bna')

    # the land-water map

    map_ = MapFromBNA(mapfile, refloat_halflife=06)  # seconds

    model = Model(time_step=timedelta(minutes=15),
                  start_time=release_time,
                  duration=timedelta(hours=1),
                  map=map_,
                  uncertain=True,
                  cache_enabled=False,
                  )

    model.movers += SimpleMover(velocity=(1., -1., 0.0))

    model.uncertain = True

    start_points = np.zeros((3, ), dtype=np.float64)
    end_points = np.zeros((3, ), dtype=np.float64)

    start_points[:] = (-127.1, 47.93, 0)
    end_points[:] = (-126.5, 48.1, 0)

    return {'model': model, 'release_start_pos': start_points,
            'release_end_pos': end_points}


@pytest.fixture(scope='function', params=['relpath', 'abspath'])
def clean_temp(request):
    temp = os.path.join(base_dir, 'temp')   # absolute path
    if os.path.exists(temp):
        shutil.rmtree(temp)

    os.mkdir(temp)    # let path get created by save_load
    if request.param == 'relpath':
        return os.path.relpath(temp)    # do save/load tests with relative path
    else:
        return temp
