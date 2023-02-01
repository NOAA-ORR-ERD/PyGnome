#!/usr/bin/env python
"""
Script to test GNOME with Chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA

NOTE: This is using the "old" C++ code, wrapped by c_GridCurrentMover
      The Chesapeake Bay OFS uses and old grid that's not compatible with
      the newer codebase. In most cases, you will want to use GridCurrentMover
"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting as gs

from gnome.basic_types import datetime_value_2d

from gnome.movers import c_GridCurrentMover


# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = datetime(2004, 12, 31, 13, 0)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = gs.Model(start_time=start_time,
                  duration=gs.days(1),
                  time_step=30 * 60,
                  uncertain=True)

    mapfile = gs.get_datafile(os.path.join(base_dir, 'ChesapeakeBay.bna'))

    print('adding the map')
    model.map = gs.MapFromBNA(mapfile, refloat_halflife=1)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = gs.Renderer(mapfile, images_dir, image_size=(800, 600),
                        output_timestep=gs.hours(2),
                        draw_ontop='forecast')
    # set the viewport to zoom in on the map:
    renderer.viewport = ((-76.5, 37.), (-75.8, 38.))
    # add the raster map, so we can see it...
    # note: this is really slow, so only use for diagnostics
    # renderer.raster_map = model.map

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_chesapeake_bay.nc')
    gs.remove_netcdf(netcdf_file)
    model.outputters += gs.NetCDFOutput(netcdf_file, which_data='all',
                                     output_timestep=gs.hours(2))

    print('adding a spill')
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill = gs.surface_point_line_spill(num_elements=1000,
                                        start_position=(-76.126872,
                                                        37.680952,
                                                        0.0),
                                        release_time=start_time)

    model.spills += spill

    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=50000)

    print('adding a wind mover:')

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (30, 0))
    series[1] = (start_time + gs.hours(23), (30, 0))

    wind = gs.Wind(timeseries=series, units='knot')

    # default is .4 radians
    w_mover = gs.PointWindMover(wind, uncertain_angle_scale=0)
    wind.extrapolation_is_allowed = True
    model.movers += w_mover

    print('adding a current mover:')
    curr_file = gs.get_datafile(os.path.join(base_dir, 'ChesapeakeBay.nc'))
    topology_file = gs.get_datafile(os.path.join(base_dir, 'ChesapeakeBay.dat'))

    # uncertain_time_delay in hours
    c_mover = c_GridCurrentMover(curr_file, topology_file,
                               uncertain_time_delay=3)
    c_mover.uncertain_along = 0  # default is .5
    # c_mover.uncertain_cross = 0  # default is .25

    model.movers += c_mover

    return model


if __name__ == "__main__":
    gs.make_images_dir()
    model = make_model()
    print("running the model")
    model.full_run()

