"""
Script to test GNOME with chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA
"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting
from gnome import utilities
from gnome.basic_types import datetime_value_2d, numerical_methods

from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, constant_wind_mover, GridCurrentMover

from gnome.outputters import Renderer
from gnome.environment.vector_field import ice_field
from gnome.movers import PyIceMover
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(1985, 1, 1, 13, 31)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(hours=96),
                  time_step=3600)

    mapfile = get_datafile(os.path.join(base_dir, 'norwegiansea.bna'))

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
#     renderer.viewport = ((-180, -90), (180, 90))
#     renderer.viewport = ((-122.9, 45.6), (-122.6, 46.0))

    print 'adding outputters'
    model.outputters += renderer

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill1 = point_line_release_spill(num_elements=20000,
                                      start_position=(-9.5,
                                                      70.5,
                                                      0.0),
                                      release_time=start_time)

    model.spills += spill1

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=100000)

    print 'adding a wind mover:'

#     model.movers += constant_wind_mover(0.5, 0, units='m/s')

    print 'adding a current mover:'

    ice, water, air = ice_field(
        'N:\\Users\\Dylan.Righi\\OutBox\\ArcticROMS\\arctic_avg2_0001_gnome.nc')
    ice_mover = PyIceMover(ice, water, air)
    model.movers += ice_mover
    ice.set_appearance(on=True)
    renderer.grids += [ice]
    renderer.delay = 25

    # curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))
    # c_mover = GridCurrentMover(curr_file)
    # model.movers += c_mover

    return model


if __name__ == "__main__":
    pd.profiler.enable()
    startTime = datetime.now()
    scripting.make_images_dir()
    model = make_model()
    print "doing full run"
    rend = model.outputters[0]
    field = rend.grids[0]
#     rend.graticule.set_DMS(True)
    for step in model:
        #         if step['step_num'] == 0:
        #             rend.set_viewport(((-165, 69.5), (-162.5, 70)))
        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    print datetime.now() - startTime
    pd.profiler.disable()
    pd.print_stats(0.2)
