#!/usr/bin/env python
"""
Script to test GNOME with chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA
"""

import os
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
from gnome.movers import RandomMover, WindMover, GridCurrentMover

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2004, 12, 31, 13, 0)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(days=1),
                  time_step=30 * 60,
                  uncertain=True)

    mapfile = get_datafile(os.path.join(base_dir, './ChesapeakeBay.bna'))

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=1)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, size=(800, 600),
                        output_timestep=timedelta(hours=2),
                        draw_ontop='uncertain')
    renderer.viewport = ((-76.5, 37.), (-75.8, 38.))

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_chesapeake_bay.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all',
                                     output_timestep=timedelta(hours=2))

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-76.126872,
                                                     37.680952, 0.0),
                                     release_time=start_time)

    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=50000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (30, 0))
    series[1] = (start_time + timedelta(hours=23), (30, 0))

    wind = Wind(timeseries=series, units='knot')

    # default is .4 radians
    w_mover = WindMover(wind, uncertain_angle_scale=0)
    model.movers += w_mover

    print 'adding a current mover:'
    curr_file = get_datafile(os.path.join(base_dir, r"./ChesapeakeBay.nc"))
    topology_file = get_datafile(os.path.join(base_dir,
                                              r"./ChesapeakeBay.dat"))

    # uncertain_time_delay in hours
    c_mover = GridCurrentMover(curr_file, topology_file,
                               uncertain_time_delay=3)
    c_mover.uncertain_along = 0  # default is .5
    #c_mover.uncertain_cross = 0  # default is .25
    model.movers += c_mover

    return model


if __name__ == "__main__":
    model = make_model()
