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

import numpy as np

import pytest

import gnome
from gnome.basic_types import datetime_value_2d

from gnome.map import MapFromBNA
from gnome.model import Model

from gnome.spill_container import SpillContainer

from gnome.movers import SimpleMover
from gnome.weatherers import Skimmer
from gnome.environment import constant_wind, Water, Waves
from gnome.utilities.remote_data import get_datafile
import gnome.array_types as gat


base_dir = os.path.dirname(__file__)

test_oil = u'ALASKA NORTH SLOPE (MIDDLE PIPELINE)'


@pytest.fixture(scope="session")
def dump():
    '''
    create dump folder for output data/files
    session scope so it is only executed the first time it is used
    We only want to create a new 'dump' folder once for each session

    Note: Takes optional dump_loc input so other test packages can import
    this as a function and use it to define their own dump directory if desired
    '''
    # dump_loc = os.path.join(request.session.fspath.strpath, 'dump')
    dump_loc = os.path.join(base_dir, 'dump')

    try:
        shutil.rmtree(dump_loc)
    except:
        pass
    try:
        os.makedirs(dump_loc)
    except:
        pass
    return dump_loc


@pytest.fixture(autouse=True)
def skip_serial(request):
    '''
    when defined in ..conftest.py, this wasn't loaded if tests are run from
    here. For now, moved this to unit_tests/conftest.py since all tests are
    contained here
    '''
    if (request.node.get_marker('serial') and
        getattr(request.config, 'slaveinput', {}).get('slaveid', 'local') !=
            'local'):
        # under xdist and serial so skip the test
        pytest.skip('serial')


def mock_sc_array_types(array_types):
    '''
    function that creates the SpillContainer's array_types attribute
    '''
    d_array_types = {}
    for array in array_types:
        if array not in d_array_types:
            try:
                d_array_types[array] = getattr(gat, array)
            except AttributeError:
                pass
        else:
            # must be a tuple of length 2
            d_array_types[array[0]] = array[1]

    return d_array_types


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
                      arr_types=None,
                      windage_range=None):
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
        spill = gnome.spill.point_line_release_spill(num_elements,
                                                     start_pos,
                                                     release_time)
    spill.units = 'g'
    spill.amount = num_elements
    if element_type is not None:
        spill.element_type = element_type

    if current_time is None:
        current_time = spill.release_time

    if arr_types is None:
        # default always has standard windage parameters required by wind_mover
        arr_types = {'windages', 'windage_range', 'windage_persist'}

    if windage_range is not None:
        spill.set('windage_range', windage_range)

    sc = SpillContainer(uncertain)
    sc.spills.add(spill)

    # used for testing so just assume there is a Windage array
    sc.prepare_for_model_run(arr_types)
    sc.release_elements(time_step, current_time)
    return sc


def get_testdata():
    '''
    define all the testdata files here
    most of these are used in multiple modules. Some are not, but let's just
    define them all in one place, ie here.
    '''
    s_data = os.path.join(base_dir, 'sample_data')
    lis = os.path.join(s_data, 'long_island_sound')
    bos = os.path.join(s_data, 'boston_data')
    dbay = os.path.join(s_data, 'delaware_bay')
    curr_dir = os.path.join(s_data, 'currents')
    tide_dir = os.path.join(s_data, 'tides')
    wind_dir = os.path.join(s_data, 'winds')
    testmap = os.path.join(base_dir, '../sample_data', 'MapBounds_Island.bna')
    bna_sample = os.path.join(s_data, 'MapBounds_2Spillable2Islands2Lakes.bna')

    data = dict()

    data['CatsMover'] = \
        {'curr': get_datafile(os.path.join(lis, 'tidesWAC.CUR')),
         'tide': get_datafile(os.path.join(lis, 'CLISShio.txt'))}
    data['ComponentMover'] = \
        {'curr': get_datafile(os.path.join(dbay, 'NW30ktwinds.cur')),
         'curr2': get_datafile(os.path.join(dbay, 'SW30ktwinds.cur')),
         'wind': get_datafile(os.path.join(dbay, 'ConstantWind.WND'))}
    data['CurrentCycleMover'] = \
        {'curr': get_datafile(os.path.join(curr_dir, 'PQBayCur.nc4')),
         'top': get_datafile(os.path.join(curr_dir, 'PassamaquoddyTOP.dat')),
         'tide': get_datafile(os.path.join(tide_dir, 'EstesHead.txt')),
         'curr_bad_file': get_datafile(os.path.join(curr_dir,
                                                    'BigCombinedwMapBad.cur'))}
    data['GridCurrentMover'] = \
        {'curr_tri': get_datafile(os.path.join(curr_dir, 'ChesBay.nc')),
         'top_tri': get_datafile(os.path.join(curr_dir, 'ChesBay.dat')),
         'curr_reg': get_datafile(os.path.join(curr_dir, 'test.cdf')),
         'curr_curv': get_datafile(os.path.join(curr_dir, 'ny_cg.nc')),
         'top_curv': get_datafile(os.path.join(curr_dir, 'NYTopology.dat')),
         'ice_curr_curv': get_datafile(os.path.join(curr_dir,
                                                    'acnfs_example.nc')),
         'ice_top_curv': get_datafile(os.path.join(curr_dir,
                                                   'acnfs_topo.dat')),
         'ptCur': get_datafile(os.path.join(curr_dir, 'ptCurNoMap.cur')),
         'grid_ts': get_datafile(os.path.join(curr_dir, 'gridcur_ts.cur')),
         'series_gridCur': get_datafile(os.path.join(curr_dir,
                                                     'gridcur_ts_hdr2.cur')),
         'series_curv': get_datafile(os.path.join(curr_dir, 'file_series',
                                                  'flist2.txt')),
         'series_top': get_datafile(os.path.join(curr_dir, 'file_series',
                                                 'HiROMSTopology.dat'))}

    data['IceMover'] = \
        {'ice_curr_curv': get_datafile(os.path.join(curr_dir,
                                                    'acnfs_example.nc')),
         'ice_top_curv': get_datafile(os.path.join(curr_dir,
                                                   'acnfs_topo.dat')),
         'ice_wind_curv': get_datafile(os.path.join(curr_dir,
                                                    'arctic_avg2_t0.nc')),
         'ice_wind_top_curv': get_datafile(os.path.join(curr_dir,
                                                        'arctic_avg2_topo.dat')
                                           )}

    # get netcdf stored in fileseries flist2.txt, gridcur_ts_hdr2
    get_datafile(os.path.join(curr_dir, 'file_series', 'hiog_file1.nc'))
    get_datafile(os.path.join(curr_dir, 'file_series', 'hiog_file2.nc'))
    get_datafile(os.path.join(curr_dir, 'gridcur_tsA.cur'))
    get_datafile(os.path.join(curr_dir, 'gridcur_tsB.cur'))

    data['GridWindMover'] = \
        {'wind_curv': get_datafile(os.path.join(wind_dir,
                                                'WindSpeedDirSubset.nc')),
         'top_curv': get_datafile(os.path.join(wind_dir,
                                               'WindSpeedDirSubsetTop.dat')),
         'wind_rect': get_datafile(os.path.join(wind_dir, 'test_wind.cdf')),
         'grid_ts': get_datafile(os.path.join(wind_dir, 'gridwind_ts.wnd')),
         'ice_wind_curv': get_datafile(os.path.join(curr_dir,
                                                    'arctic_avg2_t0.nc')),
         'ice_wind_top_curv': get_datafile(os.path.join(curr_dir,
                                                        'arctic_avg2_topo.dat')
                                           )}

    data['MapFromBNA'] = {'testmap': testmap}
    data['Renderer'] = {'bna_sample': bna_sample,
                        'bna_star': os.path.join(s_data, 'Star.bna')}
    data['GridMap'] = \
        {'curr': get_datafile(os.path.join(curr_dir, 'ny_cg.nc')),
         'BigCombinedwMap':
            get_datafile(os.path.join(curr_dir, 'BigCombinedwMap.cur')),
         }

    # following are not on server, they are part of git repo so just set the
    # path correctly
    data['timeseries'] = \
        {'wind_ts': os.path.join(s_data, 'WindDataFromGnome.WND'),
         'wind_ts_av': os.path.join(s_data, 'WindDataFromGnomeAv.WND'),
         'wind_constant': os.path.join(s_data,
                                       'WindDataFromGnomeConstantWind.WND'),
         'wind_bad_units': os.path.join(s_data,
                                        'WindDataFromGnome_BadUnits.WND'),
         'wind_cardinal': os.path.join(s_data,
                                       'WindDataFromGnomeCardinal.WND'),
         'wind_kph': os.path.join(s_data, 'WindDataFromGnomeKPH.WND'),
         'tide_shio': get_datafile(os.path.join(tide_dir, 'CLISShio.txt')),
         'tide_ossm': get_datafile(os.path.join(tide_dir, 'TideHdr.FINAL'))
         }

    # data for boston model - used for testing save files/webapi
    data['boston_data'] = \
        {'map': get_datafile(os.path.join(bos, 'MassBayMap.bna')),
         'cats_curr1': get_datafile(os.path.join(bos, 'EbbTides.cur')),
         'cats_shio': get_datafile(os.path.join(bos, 'EbbTidesShio.txt')),
         'cats_curr2': get_datafile(os.path.join(bos,
                                                 'MerrimackMassCoast.cur')),
         'cats_ossm': get_datafile(os.path.join(bos,
                                                'MerrimackMassCoastOSSM.txt')),
         'cats_curr3': get_datafile(os.path.join(bos, 'MassBaySewage.cur')),
         'component_curr1': get_datafile(os.path.join(bos, "WAC10msNW.cur")),
         'component_curr2': get_datafile(os.path.join(bos, "WAC10msSW.cur"))
         }

    data['nc'] = {'nc_output':
                  get_datafile(os.path.join(s_data, 'nc', 'test_output.nc'))}
    data['lis'] = \
        {'map': get_datafile(os.path.join(lis, 'LongIslandSoundMap.BNA')),
         'cats_curr': get_datafile(os.path.join(lis, r"LI_tidesWAC.CUR")),
         'cats_tide': get_datafile(os.path.join(lis, r"CLISShio.txt"))
         }
    return data


# create the dict here
testdata = get_testdata()


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

rq = np.array([(1, 0),
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

    uv = np.array([(0, -1),
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

    uv = np.array([(0, 1),
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
    from gnome.utilities.graphs import Graph

    return Graph(points=((1, 2, 3),
                         (2, 3, 4),
                         (3, 4, 5),),
                 labels=('x', 'F1(x)', 'F2(x)'),
                 formats=('', 'r-o', 'g->'),
                 title='Custom line styles'
                 )


@pytest.fixture(scope='module')
def wind_timeseries(rq_wind):
    dtv_rq = np.zeros((len(rq_wind['rq']), ),
                      dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(2012, 11, 06,
                            20, 10 + i, 0)
                   for i in range(len(dtv_rq))]
    dtv_rq.value = rq_wind['rq']

    dtv_uv = np.zeros((len(dtv_rq), ),
                      dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.time = dtv_rq.time
    dtv_uv.value = rq_wind['uv']

    return {'rq': dtv_rq, 'uv': dtv_uv}


@pytest.fixture(scope='module')
def wind_circ(wind_timeseries):
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
    dtv_rq = wind_timeseries['rq']
    wm = environment.Wind(timeseries=dtv_rq, format='r-theta',
                          units='meter per second')

    return {'wind': wm, 'rq': dtv_rq, 'uv': wind_timeseries['uv']}


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
    end_release_time = release_time + timedelta(hours=24)

    vps = VerticalPlumeRelease(num_elements=200,
                               start_position=(28, -78, 0.),
                               release_time=release_time,
                               end_release_time=end_release_time,
                               plume_data=get_plume_data())

    vps.plume_gen.time_step_delta = timedelta(hours=1).total_seconds()
    return Spill(vps)


@pytest.fixture(scope='function')
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
    release_time_2 = release_time + timedelta(hours=1)

    end_position = (24.0, -79.5, 1.0)
    end_release_time = datetime(2012, 1, 1, 12) + timedelta(hours=4)

    spills = [gnome.spill.point_line_release_spill(num_elements,
                                                   start_position,
                                                   release_time,
                                                   amount=10, units='l'),
              gnome.spill.point_line_release_spill(num_elements,
                                                   start_position,
                                                   release_time_2,
                                                   end_position,
                                                   end_release_time),
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


@pytest.fixture(scope='module')
def sample_model2():
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

    mapfile = os.path.join(os.path.dirname(__file__),
                           './sample_data/long_island_sound',
                           'LongIslandSoundMap.BNA')

    # the land-water map

    map_ = MapFromBNA(mapfile, refloat_halflife=06)  # seconds

    model = Model(time_step=timedelta(minutes=10),
                  start_time=release_time,
                  duration=timedelta(hours=1),
                  map=map_,
                  uncertain=True,
                  cache_enabled=False,
                  )

    # model.movers += SimpleMover(velocity=(1., -1., 0.0))

    # model.uncertain = True

    start_points = np.zeros((3, ), dtype=np.float64)
    end_points = np.zeros((3, ), dtype=np.float64)

    start_points[:] = (-72.83, 41.13, 0)
    end_points[:] = (-72.83, 41.13, 0)

    return {'model': model, 'release_start_pos': start_points,
            'release_end_pos': end_points}


@pytest.fixture(scope='function')
def sample_model_fcn():
    'sample_model with function scope'
    return sample_model()


@pytest.fixture(scope='function')
def sample_model_fcn2():
    'sample_model with function scope'
    return sample_model2()


def sample_model_weathering(sample_model_fcn,
                            oil,
                            temp=311.16,
                            num_les=10):
    model = sample_model_fcn['model']
    rel_pos = sample_model_fcn['release_start_pos']

    # update model the same way for multiple tests
    model.uncertain = False     # fixme: with uncertainty, copying spill fails!
    model.duration = timedelta(hours=4)

    et = gnome.spill.elements.floating(substance=oil)
    start_time = model.start_time + timedelta(hours=1)
    end_time = start_time + timedelta(seconds=model.time_step*3)
    spill = gnome.spill.point_line_release_spill(num_les,
                                                 rel_pos,
                                                 start_time,
                                                 end_release_time=end_time,
                                                 element_type=et,
                                                 amount=100,
                                                 units='kg')
    model.spills += spill

    # define environment objects that weatherers require
    model.environment += [constant_wind(1, 0), Water(), Waves()]

    return model


def sample_model_weathering2(sample_model_fcn2, oil, temp=311.16):
    model = sample_model_fcn2['model']
    rel_pos = sample_model_fcn2['release_start_pos']

    # update model the same way for multiple tests
    model.uncertain = False     # fixme: with uncertainty, copying spill fails!
    model.duration = timedelta(hours=24)

    et = gnome.spill.elements.floating(substance=oil)
    start_time = model.start_time
    end_time = start_time
    spill = gnome.spill.point_line_release_spill(10,
                                                 rel_pos,
                                                 start_time,
                                                 end_release_time=end_time,
                                                 element_type=et,
                                                 amount=10000,
                                                 units='kg')
    model.spills += spill

    return model


@pytest.fixture(scope='function')
def saveloc_(tmpdir, request):
    '''
    create a temporary save location
    '''
    name = 'save_' + request.function.func_name
    name = tmpdir.mkdir(name).strpath

    return name


'''
Default properties for CyCurrentMover base class - double check cython
derived classes are getting/setting cython base class properties correctly
'''


@pytest.fixture(scope='function')
def CyCurrentMover_props():
    'gives the property names and default values for CyCurrentMover base class'
    default_prop = (('uncertain_duration', 172800),
                    ('uncertain_time_delay', 0),
                    ('up_cur_uncertain', 0.3),
                    ('down_cur_uncertain', -0.3),
                    ('right_cur_uncertain', 0.1),
                    ('left_cur_uncertain', -0.1))

    return default_prop
