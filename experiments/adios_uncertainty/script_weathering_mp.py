import os

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

from datetime import datetime, timedelta
import multiprocessing

import numpy
np = numpy

from gnome import scripting
from gnome.basic_types import datetime_value_2d
from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind, Water, Tide

from gnome.spill import point_line_release_spill
from gnome.spill.elements import floating_weathering

from gnome.movers import RandomMover, WindMover, CatsMover
from gnome.weatherers import Evaporation, Dispersion, Skimmer, Burn

from gnome.outputters import WeatheringOutput

from multi_model_broadcast import ModelBroadcaster

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
    print 'et =', et._substance._r_oil.sara_fractions
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-72.419992,
                                                     41.202120, 0.0),
                                     release_time=start_time,
                                     amount=1000,
                                     units='kg',
                                     element_type=et)
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

    wind = Wind(timeseries=series, units='m/s')
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

    return model


if __name__ == '__main__':
    scripting.make_images_dir()

    model = make_model()

    model_broadcaster = ModelBroadcaster(model, 2)
    print 'step results:'
    pp.pprint(model_broadcaster.cmd('step', {}))

    model_broadcaster.stop()

    # model.full_run(logger=True)
    # post_run(model)
