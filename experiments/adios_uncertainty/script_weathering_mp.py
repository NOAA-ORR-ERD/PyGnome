import os

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

from datetime import datetime, timedelta

import numpy
np = numpy

import zmq

from gnome import scripting
from gnome.basic_types import datetime_value_2d
from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.maps import MapFromBNA
from gnome.environment import Wind, Water, Tide

from gnome.spills import point_line_release_spill
from gnome.spills.elements import floating_weathering

from gnome.movers import RandomMover, PointWindMover, CatsMover
from gnome.weatherers import Evaporation, Dispersion, Skimmer, Burn

from gnome.outputters import WeatheringOutput

from gnome.multi_model_broadcast import ModelBroadcaster

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2012, 9, 15, 12, 0)
    mapfile = get_datafile(os.path.join(base_dir, './LongIslandSoundMap.BNA'))

    gnome_map = MapFromBNA(mapfile, refloat_halflife=6)  # hours

    # # the image output renderer
    # global renderer

    # one hour timestep
    model = Model(start_time=start_time,
                  duration=timedelta(hours=48), time_step=3600,
                  map=gnome_map, uncertain=False, cache_enabled=False)

    print 'adding a spill'
    et = floating_weathering(substance='FUEL OIL NO.6')
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-72.419992,
                                                     41.202120, 0.0),
                                     release_time=start_time,
                                     amount=1000,
                                     units='kg',
                                     element_type=et)
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
    model.movers += PointWindMover(wind)

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

    return model


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
    print 'main(): stopped broadcaster.'
