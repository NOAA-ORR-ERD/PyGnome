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

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2015, 9, 24, 3, 0)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(hours = 48),
                  time_step=3600)

    mapfile = get_datafile(os.path.join(base_dir, 'PNW.bna'))

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=1)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, size=(800, 600),
                        output_timestep=timedelta(hours=1))
    renderer.viewport = ((-124.25, 47.5), (-122.0, 48.70))

    print 'adding outputters'
    model.outputters += renderer

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill = point_line_release_spill(num_elements=500000,
                                     start_position=(-123.5,
                                                     48.25,
                                                     0.0),
                                     release_time=start_time)

    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=500000)

    print 'adding a wind mover:'
   
    model.movers += constant_wind_mover(13, 270, units='m/s')

    print 'adding a current mover:'
    #curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))

    # uncertain_time_delay in hours
    #c_mover = GridCurrentMover(curr_file)
    # c_mover.uncertain_cross = 0  # default is .25

    #model.movers += c_mover

    return model


if __name__ == "__main__":
    startTime = datetime.now()
    scripting.make_images_dir()
    model = make_model()
    print "doing full run"
    for step in model:
        # print step
        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    print datetime.now() - startTime