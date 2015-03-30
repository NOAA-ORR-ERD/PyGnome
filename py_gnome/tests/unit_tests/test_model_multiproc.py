import os

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

from datetime import datetime, timedelta

from pytest import raises, mark
import sys

import numpy
np = numpy

from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind, Water, Tide

from gnome.spill import point_line_release_spill

from gnome.movers import RandomMover, WindMover, CatsMover
from gnome.weatherers import Evaporation, Dispersion, Burn, Skimmer

from gnome.outputters import WeatheringOutput, GeoJson

from gnome.multi_model_broadcast import ModelBroadcaster
from conftest import testdata, test_oil

pytestmark = [mark.skipif("sys.platform=='win32'", reason="skip on windows"),
              mark.serial]


def make_model(uncertain=False,
               geojson_output=False):
    print 'initializing the model'

    start_time = datetime(2012, 9, 15, 12, 0)
    mapfile = testdata["lis"]["map"]

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
                                     substance=test_oil,
                                     units='kg')
    spill.amount_uncertainty_scale = 1.0
    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=500000, uncertain_factor=2)

    print 'adding a wind mover:'
    series = np.zeros((5, ), dtype=datetime_value_2d)
    series[0] = (start_time, (20, 45))
    series[1] = (start_time + timedelta(hours=18), (20, 90))
    series[2] = (start_time + timedelta(hours=30), (20, 135))
    series[3] = (start_time + timedelta(hours=42), (20, 180))
    series[4] = (start_time + timedelta(hours=54), (20, 225))

    wind = Wind(timeseries=series, units='m/s',
                speed_uncertainty_scale=0.05)
    model.movers += WindMover(wind)

    print 'adding a cats mover:'
    c_mover = CatsMover(testdata["lis"]["cats_curr"],
                        tide=Tide(testdata["lis"]["cats_tide"]))
    model.movers += c_mover

    model.environment += c_mover.tide

    print 'adding Weatherers'
    rel_time = model.spills[0].get('release_time')
    skim_start = rel_time + timedelta(hours=4)
    amount = spill.amount
    units = spill.units

    # define skimmer/burn cleanup options
    skimmer = Skimmer(0.5*amount, units=units, efficiency=0.3,
                      active_start=skim_start,
                      active_stop=skim_start + timedelta(hours=4))
    # thickness = 1m so area is just 20% of volume
    volume = spill.get_mass()/spill.get('substance').get_density()
    burn = Burn(0.2 * volume, 1.0,
                active_start=skim_start)

    water_env = Water(311.15)
    model.environment += water_env
    model.weatherers += [Evaporation(water_env, wind),
                         Dispersion(),
                         burn,]
                         #skimmer]

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
    model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'up'),
                                         ('down', 'up'))
    assert len(model_broadcaster.tasks) == 4
    model_broadcaster.stop()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    assert len(model_broadcaster.tasks) == 9
    model_broadcaster.stop()


def test_uncertainty_array_indexing():
    model = make_model()

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    print '\nGetting time & spill values for just the (down, down) model:'
    res = model_broadcaster.cmd('get_wind_timeseries', {}, ('down', 'down'))
    assert np.allclose([r[0] for r in res], 17.449237)

    res = model_broadcaster.cmd('get_spill_amounts', {}, ('down', 'down'))
    assert np.isclose(res[0], 333.33333)

    print '\nGetting time & spill values for just the (up, up) model:'
    res = model_broadcaster.cmd('get_wind_timeseries', {}, ('up', 'up'))
    print 'get_wind_timeseries:'
    pp.pprint(res)
    assert np.allclose([r[0] for r in res], 20.166224)

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


def test_spill_containers_have_uncertainty_off():
    model = make_model(uncertain=True)

    model_broadcaster = ModelBroadcaster(model,
                                         ('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))
    print '\nSpill results:'
    res = model_broadcaster.cmd('get_spill_container_uncertainty', {})
    print [r for r in res]
    assert not any([r for r in res])

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
