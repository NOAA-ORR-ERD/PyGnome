#!/usr/bin/env python
"""
Script to test GNOME with long island sound data
"""

import os
import shutil
from pathlib import Path

import numpy as np

import gnome.scripting as gs
from gnome.basic_types import datetime_value_2d


# define base directory
base_dir = Path(__file__).parent


def make_model(images_dir=base_dir /'images'):
    print('initializing the model')

    start_time = "2012-09-15 12:00"
    mapfile = gs.get_datafile(base_dir / 'LongIslandSoundMap.BNA')

    gnome_map = gs.MapFromBNA(mapfile, refloat_halflife=6)  # hours

    # # the image output renderer
    # global renderer

    # one hour timestep
    model = gs.Model(start_time=start_time,
                     duration=gs.hours(48),
                     time_step=3600,
                     map=gnome_map,
                     uncertain=True,
                     cache_enabled=True)

    print('adding outputters')
    model.outputters += gs.Renderer(mapfile, images_dir, image_size=(800, 600))

    netcdf_file = base_dir / 'script_long_island.nc'
    gs.remove_netcdf(netcdf_file)

    model.outputters += gs.NetCDFOutput(netcdf_file, which_data='all')

    print('adding a spill')
    spill = gs.surface_point_line_spill(num_elements=1000,
                                        start_position=(-72.419992,
                                                        41.202120, 0.0),
                                        release_time=start_time)
    model.spills += spill

    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=500000, uncertain_factor=2)

    print('adding a wind mover:')
    series = np.zeros((5, ), dtype=datetime_value_2d)
    start_time = gs.asdatetime(start_time)
    series[0] = (start_time, (10, 45))
    series[1] = (start_time + gs.hours(18), (10, 90))
    series[2] = (start_time + gs.hours(30), (10, 135))
    series[3] = (start_time + gs.hours(42), (10, 180))
    series[4] = (start_time + gs.hours(54), (10, 225))

    wind = gs.Wind(timeseries=series, units='m/s')
    model.movers += gs.PointWindMover(wind)

    print('adding a cats mover:')
    curr_file = gs.get_datafile(base_dir / 'LI_tidesWAC.CUR')
    tide_file = gs.get_datafile(base_dir / 'CLISShio.txt')

    c_mover = gs.CatsMover(str(curr_file), tide=gs.Tide(str(tide_file)))

    model.movers += c_mover
    model.environment += c_mover.tide

    print('viewport is:', [o.viewport
                           for o in model.outputters
                           if isinstance(o, gs.Renderer)])

    return model


def post_run(model):

    # create a place for test images (cleaning out any old ones)
    images_dir = base_dir / 'images_2'

    if images_dir.is_dir():
        shutil.rmtree(images_dir)

    os.mkdir(images_dir)

    renderers = [o for o in model.outputters
                 if isinstance(o, gs.Renderer)]

    print('re-rendering images')
    if renderers:
        renderer = renderers[0]

        renderer.images_dir = images_dir
        renderer.viewport = ((-72.75, 41.1), (-72.34, 41.3))

        renderer.prepare_for_model_run(model.start_time)

        for step_num in range(model.num_time_steps):
            print('writing image:')
            image_info = renderer.write_output(step_num)
            print('image written:', image_info)

        print('viewport is:', renderer.viewport)
    else:
        print('No Renderers available!!!')


if __name__ == '__main__':
    gs.make_images_dir()

    model = make_model()

    model.full_run()
    post_run(model)
