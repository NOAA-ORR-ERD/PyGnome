#!/usr/bin/env python
"""
Script to test GNOME with all weatherers and response options
"""

import os
import shutil
from datetime import datetime, timedelta

import numpy
np = numpy

from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput
from gnome.outputters import WeatheringOutput

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (Emulsification,
                              Evaporation,
                              NaturalDispersion,
                              ChemicalDispersion,
                              Burn,
                              Skimmer,
                              WeatheringData)

from gnome.persist import load
# define base directory
base_dir = os.path.dirname(__file__)


water = Water(280.928)
wind = constant_wind(20., 117, 'knots')
waves = Waves(wind, water)

def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2015, 5, 14, 0, 0)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(days=1.75),
                  time_step=60 * 60,
                  uncertain=True)

#     mapfile = get_datafile(os.path.join(base_dir, './ak_arctic.bna'))
# 
#     print 'adding the map'
#     model.map = MapFromBNA(mapfile, refloat_halflife=1)  # seconds
# 
#     # draw_ontop can be 'uncertain' or 'forecast'
#     # 'forecast' LEs are in black, and 'uncertain' are in red
#     # default is 'forecast' LEs draw on top
#     renderer = Renderer(mapfile, images_dir, size=(800, 600),
#                         output_timestep=timedelta(hours=2),
#                         draw_ontop='forecast')
# 
#     print 'adding outputters'
#     model.outputters += renderer

    model.outputters += WeatheringOutput()

    netcdf_file = os.path.join(base_dir, 'script_weatherers.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all',
                                     output_timestep=timedelta(hours=1))

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    end_time = start_time + timedelta(hours=24)
    spill = point_line_release_spill(num_elements=100,
                                     start_position=(-164.791878561,
                                                     69.6252597267, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time,
                                     amount=1000,
                                     substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE)',
                                     units='bbl')

    # set bullwinkle to .303 to cause mass goes to zero bug at 24 hours (when continuous release ends)
    spill.element_type._substance._bullwinkle = .303
    model.spills += spill

    print 'adding a RandomMover:'
    #model.movers += RandomMover(diffusion_coef=50000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (20, 0))
    series[1] = (start_time + timedelta(hours=23), (20, 0))

    wind2 = Wind(timeseries=series, units='knot')

    w_mover = WindMover(wind)
    model.movers += w_mover

    print 'adding weatherers and cleanup options:'

    # define skimmer/burn cleanup options
    skim1_start = start_time + timedelta(hours=15.58333)
    skim2_start = start_time + timedelta(hours=16)
    units = spill.units
    skimmer1 = Skimmer(80, units=units, efficiency=0.36,
                      active_start=skim1_start,
                      active_stop=skim1_start + timedelta(hours=8))
    skimmer2 = Skimmer(120, units=units, efficiency=0.2,
                      active_start=skim2_start,
                      active_stop=skim2_start + timedelta(hours=12))

    burn_start = start_time + timedelta(hours=36)
    burn = Burn(1000., .1,
                active_start=burn_start, efficiency=.2)

    chem_start = start_time + timedelta(hours=24)
    c_disp = ChemicalDispersion(0.5, efficiency=0.4,
                                active_start=chem_start,
                                active_stop=chem_start + timedelta(hours=8))


    model.environment += [Water(280.928), wind,  waves]

    model.weatherers += Evaporation(water,wind)
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion(waves,water)
    model.weatherers += skimmer1
    model.weatherers += skimmer2
    model.weatherers += burn
    model.weatherers += c_disp

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    model.full_run()
