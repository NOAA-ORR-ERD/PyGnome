#!/usr/bin/env python
"""
Script to test GNOME with long island sound data
"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting
from gnome.basic_types import datetime_value_2d
from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.maps import MapFromBNA
from gnome.environment import Wind, Tide
from gnome.spills import surface_point_line_spill
from gnome.movers import RandomMover, PointWindMover, CurrentCycleMover

from gnome.outputters import (Renderer,
                              NetCDFOutput,
                              )

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = datetime(2014, 6, 9, 0, 0)
    mapfile = get_datafile(os.path.join(base_dir, 'PassamaquoddyMap.bna'))

    gnome_map = MapFromBNA(mapfile, refloat_halflife=1)  # hours

    # # the image output renderer
    # global renderer

    # one hour timestep
    model = Model(start_time=start_time,
                  duration=timedelta(hours=24), time_step=360,
                  map=gnome_map, uncertain=False, cache_enabled=True)

    print('adding outputters')
    renderer = Renderer(mapfile, images_dir, image_size=(800, 600),
                        # output_timestep=timedelta(hours=1),
                        draw_ontop='uncertain')
    renderer.viewport = ((-67.15, 45.), (-66.9, 45.2))

    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_passamaquoddy.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print('adding a spill')
    spill = surface_point_line_spill(num_elements=1000,
                                     start_position=(-66.991344, 45.059316,
                                                     0.0),
                                     release_time=start_time)
    model.spills += spill

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=30000, uncertain_factor=2)

    print('adding a wind mover:')
    series = np.zeros((5, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 90))
    series[1] = (start_time + timedelta(hours=18), (5, 180))
    series[2] = (start_time + timedelta(hours=30), (5, 135))
    series[3] = (start_time + timedelta(hours=42), (5, 180))
    series[4] = (start_time + timedelta(hours=54), (5, 225))

    wind = Wind(timeseries=series, units='m/s')
    model.movers += PointWindMover(wind)

    print('adding a current mover:')
    curr_file = get_datafile(os.path.join(base_dir, 'PQBayCur.nc4'))
    topology_file = get_datafile(os.path.join(base_dir, 'PassamaquoddyTOP.dat')
                                 )
    tide_file = get_datafile(os.path.join(base_dir, 'EstesHead.txt'))

    cc_mover = CurrentCycleMover(curr_file, topology_file,
                                 tide=Tide(tide_file))

    model.movers += cc_mover
    model.environment += cc_mover.tide

    print('viewport is:', [o.viewport
                           for o in model.outputters
                           if isinstance(o, Renderer)])

    return model


if __name__ == '__main__':
    scripting.make_images_dir()
    model = make_model()
    model.full_run()
