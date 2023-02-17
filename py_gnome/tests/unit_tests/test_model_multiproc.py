
import os
import sys
import time
import traceback

from datetime import datetime, timedelta

import pytest

import numpy as np

from gnome import scripting
from gnome.basic_types import datetime_value_2d
from gnome.utilities.inf_datetime import InfDateTime

from gnome.model import Model

from gnome.maps import MapFromBNA
from gnome.environment import Wind, Water, Waves, Tide

from gnome.spills import surface_point_line_spill

from gnome.movers import RandomMover, PointWindMover, CatsMover
from gnome.weatherers import Evaporation, ChemicalDispersion, Burn, Skimmer

from gnome.outputters import WeatheringOutput, TrajectoryGeoJsonOutput

try:
    import pyzmq
    import tornado
except ImportError:
    print('Could not import Model Broadcaster -- it needs pyzmq and tornado')
else:
    from gnome.multi_model_broadcast import ModelBroadcaster

from .conftest import testdata, test_oil

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2, width=120)


pytestmark = pytest.mark.skipif(True, reason="doesn't work until we update "
                                "Cython __cinit__ code")

# pytestmark = pytest.mark.skipif((os.name == 'nt'),
#                                 reason="skip on windows")


def make_model(uncertain=False,
               geojson_output=False):

    start_time = datetime(2012, 9, 15, 12, 0)
    mapfile = testdata["lis"]["map"]

    gnome_map = MapFromBNA(mapfile, refloat_halflife=6)  # hours

    # # the image output renderer
    # global renderer

    # one hour timestep
    model = Model(start_time=start_time,
                  duration=timedelta(hours=48), time_step=3600,
                  map=gnome_map, uncertain=uncertain, cache_enabled=False)

    spill = surface_point_line_spill(num_elements=1000,
                                     start_position=(-72.419992,
                                                     41.202120, 0.0),
                                     release_time=start_time,
                                     amount=1000,
                                     substance=test_oil,
                                     units='kg')
    spill.amount_uncertainty_scale = 1.0
    model.spills += spill

    model.movers += RandomMover(diffusion_coef=500000, uncertain_factor=2)

    print('adding a wind mover:')
    series = np.zeros((5, ), dtype=datetime_value_2d)
    series[0] = (start_time, (20, 45))
    series[1] = (start_time + timedelta(hours=18), (20, 90))
    series[2] = (start_time + timedelta(hours=30), (20, 135))
    series[3] = (start_time + timedelta(hours=42), (20, 180))
    series[4] = (start_time + timedelta(hours=54), (20, 225))

    wind = Wind(timeseries=series, units='m/s',
                speed_uncertainty_scale=0.05)
    model.movers += PointWindMover(wind)

    print('adding a cats mover:')
    c_mover = CatsMover(testdata["lis"]["cats_curr"],
                        tide=Tide(testdata["lis"]["cats_tide"]))
    model.movers += c_mover

    model.environment += c_mover.tide

    print('adding Weatherers')
    rel_time = model.spills[0].release_time
    skim_start = rel_time + timedelta(hours=4)
    amount = spill.amount
    units = spill.units

    water_env = Water(311.15)
    waves = Waves(wind, water_env)
    model.environment += water_env

    # define skimmer/burn cleanup options
    skimmer = Skimmer(0.3 * amount,
                      units=units,
                      efficiency=0.3,
                      active_range=(skim_start,
                                    skim_start + timedelta(hours=4)))
    # thickness = 1m so area is just 20% of volume
    volume = spill.get_mass() / spill.substance.density_at_temp()
    burn = Burn(0.2 * volume, 1.0,
                active_range=(skim_start, InfDateTime('inf')),
                efficiency=.9)
    c_disp = ChemicalDispersion(0.1, waves=waves, efficiency=0.5,
                                active_range=(skim_start,
                                              skim_start + timedelta(hours=1)))

    model.weatherers += [Evaporation(water_env, wind),
                         c_disp,
                         burn,
                         skimmer]

    print('adding outputters')
    model.outputters += WeatheringOutput()

    if geojson_output:
        model.outputters += TrajectoryGeoJsonOutput()

    return model


@pytest.mark.timeout(30)
def test_init():
    model = make_model()

    with pytest.raises(TypeError):
        # no uncertainty arguments
        ModelBroadcaster(model)

    with pytest.raises(TypeError):
        # no spill amount uncertainties
        ModelBroadcaster(model,
                         ('down', 'normal', 'up'))

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        assert hasattr(model_broadcaster, 'id')
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_uncertainty_array_size():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down',),
                                         ('down',))

    try:
        assert len(model_broadcaster.tasks) == 1
    finally:
        model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'up'),
                                         ('down', 'up'))

    try:
        assert len(model_broadcaster.tasks) == 4
    finally:
        model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        assert len(model_broadcaster.tasks) == 9
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_uncertainty_array_indexing():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        res = model_broadcaster.cmd('get_wind_timeseries', {},
                                    ('down', 'down'))
        assert np.allclose([r[0] for r in res], 17.449237)

        res = model_broadcaster.cmd('get_spill_amounts', {}, ('down', 'down'))
        assert np.isclose(res[0], 333.33333)

        res = model_broadcaster.cmd('get_wind_timeseries', {}, ('up', 'up'))
        assert np.allclose([r[0] for r in res], 20.166224)

        res = model_broadcaster.cmd('get_spill_amounts', {}, ('up', 'up'))
        assert np.isclose(res[0], 1666.66666)
    finally:
        model_broadcaster.stop()


def is_none(results):
    'evaluate the results of a multiproc command that has timed out'
    return results is None


def is_valid(results):
    'evaluate the results of a multiproc command that successfully returned'
    return len(results) == 9


@pytest.mark.slow
@pytest.mark.parametrize(('secs', 'timeout', 'expected_runtime', 'valid_func'),
                         [(5, None, 5, is_valid),
                          (11, None, 10, is_none),
                          (4, 5, 4, is_valid),
                          (5, 4, 4, is_none)
                          ])
def test_timeout(secs, timeout, expected_runtime, valid_func):
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nsleeping for {} secs...'.format(secs))
        if timeout is None:
            begin = time.time()
            res = model_broadcaster.cmd('sleep', {'secs': secs})
            end = time.time()
        else:
            begin = time.time()
            res = model_broadcaster.cmd('sleep', {'secs': secs},
                                        timeout=timeout)
            end = time.time()

        rt = end - begin

        # runtime duraton should be either:
        # - the expected response time plus a bit of overhead
        # - the expected timeout plus a bit of overhead
        print('runtime: ', rt)
        assert rt >= expected_runtime
        assert rt < expected_runtime + (expected_runtime * 0.06)

        assert valid_func(res)
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
def test_timeout_2_times():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        #
        # First, we set a short timeout for a command, but a shorter command.
        # The command should succeed
        #
        secs, timeout, expected_runtime = 4, 5, 4
        print('\nsleeping for {} secs...'.format(secs))

        begin = time.time()
        res = model_broadcaster.cmd('sleep', {'secs': secs}, timeout=timeout)
        end = time.time()

        rt = end - begin

        assert rt >= expected_runtime
        assert rt < expected_runtime + (expected_runtime * 0.06)
        assert is_valid(res)

        #
        # Next, run a command with no timeout specified.  The timeout should
        # have reverted back to the default, and the command should succeed.
        #
        secs, expected_runtime = 9, 9
        print('\nsleeping for {} secs...'.format(secs))

        begin = time.time()
        res = model_broadcaster.cmd('sleep', {'secs': secs})
        end = time.time()

        rt = end - begin

        assert rt >= expected_runtime
        assert rt < expected_runtime + (expected_runtime * 0.06)
        assert is_valid(res)

    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_rewind():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nRewind results:')
        res = model_broadcaster.cmd('rewind', {})

        assert len(res) == 9
        assert all([r is None for r in res])
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_step():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nStep results:')
        res = model_broadcaster.cmd('step', {})
        assert len(res) == 9
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_full_run():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nNumber of time steps:')
        num_steps = model_broadcaster.cmd('num_time_steps', {})
        assert len(num_steps) == 9

        # all models have the same number of steps
        assert len(set(num_steps)) == 1

        print('\nStep results:')
        res = model_broadcaster.cmd('full_run', {})
        assert len(res) == 9

        for n, r in zip(num_steps, res):
            assert len(r) == n
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_cache_dirs():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nCache directory results:')
        res = model_broadcaster.cmd('get_cache_dir', {})

        assert all([os.path.isdir(d) for d in res])
        assert len(set(res)) == 9  # all dirs should be unique
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_spill_containers_have_uncertainty_off():
    model = make_model(uncertain=True)

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        print('\nSpill results:')
        res = model_broadcaster.cmd('get_spill_container_uncertainty', {})
        print([r for r in res])
        assert not any([r for r in res])
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_weathering_output_only():
    model = make_model(geojson_output=True)

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    try:
        res = model_broadcaster.cmd('get_outputters', {})

        assert not [o for r in res for o in r
                    if not isinstance(o, WeatheringOutput)]

        res = model_broadcaster.cmd('step', {})

        assert len(res) == 9

        assert [list(r.keys()) for r in res
                if ('step_num' in r and
                    'valid' in r and
                    'WeatheringOutput' in r)]
    finally:
        model_broadcaster.stop()


@pytest.mark.slow
@pytest.mark.timeout(10)
@pytest.mark.xfail()
def test_child_exception():
    '''
        This one is a bit tricky.  We would like to simulate an exception
        by making the spill.amount a None value and then instantiating
        our broadcaster.  This is expected to raise a TypeError based on
        the current codebase, but could change.

        We would like to get the exception raised in the child process and
        re-raise it in the broadcaster parent process, complete with good
        traceback information.
    '''
    model = make_model(geojson_output=True)

    model.spills[0].amount = None
    print('amount:', model.spills[0].amount)

    try:
        _model_broadcaster = ModelBroadcaster(model,
                                              ('down', 'normal', 'up'),
                                              ('down', 'normal', 'up'))
    except Exception as e:
        #assert type(e) == TypeError
        assert type(e) == ValueError

        exc_type, exc_value, exc_traceback = sys.exc_info()
        fmt = traceback.format_exception(exc_type, exc_value, exc_traceback)

        last_file_entry = [l for l in fmt if l.startswith('  File ')][-1]
        last_file = last_file_entry.split('"')[1]

        assert os.path.basename(last_file) == 'spill.py'


if __name__ == '__main__':
    scripting.make_images_dir()

    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    print('\nStep results:')
    pp.pprint(model_broadcaster.cmd('step', {}))

    print('\nGetting wind timeseries for all models:')
    pp.pprint(model_broadcaster.cmd('get_wind_timeseries', {}))

    print('\nGetting spill amounts for all models:')
    pp.pprint(model_broadcaster.cmd('get_spill_amounts', {}))

    print('\nGetting time & spill values for just the (down, down) model:')
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('down', 'down')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('down', 'down')),
               ))

    print('\nGetting time & spill values for just the (normal, normal) model:')
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('normal', 'normal')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('normal', 'normal')),
               ))

    print('\nGetting time & spill values for just the (up, up) model:')
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('up', 'up')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('up', 'up')),
               ))

    model_broadcaster.stop()
