import os

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

from datetime import datetime, timedelta

import pytest
from pytest import raises

import numpy
np = numpy

from gnome import scripting
from gnome.basic_types import datetime_value_2d
from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind, Water, Tide

from gnome.spill import point_line_release_spill

from gnome.movers import RandomMover, WindMover, CatsMover
from gnome.weatherers import Evaporation, Dispersion, Skimmer, Burn

from gnome.outputters import WeatheringOutput, GeoJson

from gnome.multi_model_broadcast import ModelBroadcaster

# define base directory
base_dir = os.path.join(os.path.dirname(__file__),
                        'sample_data',
                        'long_island_sound')


def make_model(images_dir=os.path.join(base_dir, 'images'),
               uncertain=False,
               geojson_output=False):
    print 'initializing the model'

    start_time = datetime(2012, 9, 15, 12, 0)
    mapfile = get_datafile(os.path.join(base_dir, 'LongIslandSoundMap.BNA'))

    gnome_map = MapFromBNA(mapfile, refloat_halflife=6)  # hours

    # # the image output renderer
    # global renderer

    # one hour timestep
    model = Model(start_time=start_time,
                  duration=timedelta(hours=48), time_step=3600,
                  map=gnome_map, uncertain=uncertain, cache_enabled=False)

    print 'adding a spill'
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-72.419992,
                                                     41.202120, 0.0),
                                     release_time=start_time,
                                     amount=1000,
                                     units='kg')
    spill.amount_uncertainty_scale = 1.0
    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=500000, uncertain_factor=2)

    print 'adding a wind mover:'
    series = np.zeros((5, ), dtype=datetime_value_2d)
    series[0] = (start_time, (10, 45))
    series[1] = (start_time + timedelta(hours=18), (10, 90))
    series[2] = (start_time + timedelta(hours=30), (10, 135))
    series[3] = (start_time + timedelta(hours=42), (10, 180))
    series[4] = (start_time + timedelta(hours=54), (10, 225))

    wind = Wind(timeseries=series, units='m/s',
                speed_uncertainty_scale=0.5)
    model.movers += WindMover(wind)

    print 'adding a cats mover:'
    curr_file = get_datafile(os.path.join(base_dir, r"./LI_tidesWAC.CUR"))
    tide_file = get_datafile(os.path.join(base_dir, r"./CLISShio.txt"))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file))
    model.movers += c_mover

    model.environment += c_mover.tide

    print 'adding Weatherers'
    water_env = Water(311.15)
    model.environment += water_env
    model.weatherers += [Evaporation(water_env, wind),
                         Dispersion(),
                         Burn(),
                         Skimmer()]

    print 'adding outputters'
    model.outputters += WeatheringOutput()

    if geojson_output:
        model.outputters += GeoJson()

    return model


def test_init():
    model = make_model()

    with raises(TypeError):
        ModelBroadcaster(model)

    with raises(TypeError):
        ModelBroadcaster(model,
                         ('down', 'normal', 'up'))

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    assert hasattr(model_broadcaster, 'id')

    model_broadcaster.stop()


def test_uncertainty_array_size():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down',),
                                         ('down',))
    assert len(model_broadcaster.tasks) == 1
    assert len(model_broadcaster.results) == 1
    model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'up'),
                                         ('down', 'up'))
    assert len(model_broadcaster.tasks) == 4
    assert len(model_broadcaster.results) == 4
    model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    assert len(model_broadcaster.tasks) == 9
    assert len(model_broadcaster.results) == 9
    model_broadcaster.stop()


def test_uncertainty_array_indexing():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    print '\nGetting time & spill values for just the (down, down) model:'
    res = model_broadcaster.cmd('get_wind_timeseries', {}, ('down', 'down'))
    assert np.allclose([r[0] for r in res], 6.010577)

    res = model_broadcaster.cmd('get_spill_amounts', {}, ('down', 'down'))
    assert np.isclose(res[0], 333.33333)

    print '\nGetting time & spill values for just the (up, up) model:'
    res = model_broadcaster.cmd('get_wind_timeseries', {}, ('up', 'up'))
    assert np.allclose([r[0] for r in res], 13.989423)

    res = model_broadcaster.cmd('get_spill_amounts', {}, ('up', 'up'))
    assert np.isclose(res[0], 1666.66666)

    model_broadcaster.stop()


def test_rewind():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nRewind results:'
    res = model_broadcaster.cmd('rewind', {})

    assert len(res) == 9
    assert all([r is None for r in res])

    model_broadcaster.stop()


def test_step():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nStep results:'
    res = model_broadcaster.cmd('step', {})
    assert len(res) == 9

    model_broadcaster.stop()


def test_full_run():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nNumber of time steps:'
    num_steps = model_broadcaster.cmd('num_time_steps', {})
    assert len(num_steps) == 9
    assert len(set(num_steps)) == 1  # all models have the same number of steps

    print '\nStep results:'
    res = model_broadcaster.cmd('full_run', {})
    assert len(res) == 9

    for n, r in zip(num_steps, res):
        assert len(r) == n

    model_broadcaster.stop()


def test_cache_dirs():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nCache directory results:'
    res = model_broadcaster.cmd('get_cache_dir', {})

    assert all([os.path.isdir(d) for d in res])
    assert len(set(res)) == 9  # all dirs should be unique

    model_broadcaster.stop()


def test_spills():
    model = make_model(uncertain=True)

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nSpill results:'
    res = model_broadcaster.cmd('get_spills', {})
    assert not any([r.uncertain for r in res])

    model_broadcaster.stop()


def test_weathering_output_only():
    model = make_model(geojson_output=True)

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    print '\nOutputter results:'
    res = model_broadcaster.cmd('get_outputters', {})

    assert not [o for r in res for o in r
                if not isinstance(o, WeatheringOutput)]

    print '\nStep results:'
    res = model_broadcaster.cmd('step', {})

    assert len(res) == 9
    assert [r.keys() for r in res
            if len(r.keys()) == 1
            and 'WeatheringOutput' in r]

    model_broadcaster.stop()


if __name__ == '__main__':
    scripting.make_images_dir()

    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    print '\nStep results:'
    pp.pprint(model_broadcaster.cmd('step', {}))

    print '\nGetting wind timeseries for all models:'
    pp.pprint(model_broadcaster.cmd('get_wind_timeseries', {}))

    print '\nGetting spill amounts for all models:'
    pp.pprint(model_broadcaster.cmd('get_spill_amounts', {}))

    print '\nGetting time & spill values for just the (down, down) model:'
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('down', 'down')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('down', 'down')),
               ))

    print '\nGetting time & spill values for just the (normal, normal) model:'
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('normal', 'normal')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('normal', 'normal')),
               ))

    print '\nGetting time & spill values for just the (up, up) model:'
    pp.pprint((model_broadcaster.cmd('get_wind_timeseries', {},
                                     ('up', 'up')),
               model_broadcaster.cmd('get_spill_amounts', {},
                                     ('up', 'up')),
               ))

    model_broadcaster.stop()

    # model.full_run(logger=True)
    # post_run(model)
