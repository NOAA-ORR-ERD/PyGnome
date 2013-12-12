#!/usr/bin/env python

"""
Script to test GNOME with long island sound data

"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind, Tide

from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome.utilities.remote_data import get_datafile
from gnome import scripting

# define base directory

base_dir = os.path.dirname(__file__)


# global renderer

def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2012, 9, 15, 12, 0)

    mapfile = get_datafile(os.path.join(base_dir,
                           './LongIslandSoundMap.BNA'))

    gnome_map = gnome.map.MapFromBNA(mapfile, refloat_halflife=6)  # hours

    # # the image output renderer
    # global renderer

    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800,
            600))

    # renderer.viewport = ((-72.75, 41.1),(-72.34, 41.3))

    model = gnome.model.Model(  # one hour in seconds
        start_time=start_time,
        duration=timedelta(hours=48),
        time_step=3600,
        map=gnome_map,
        uncertain=True,
        cache_enabled=True,
        )

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_long_island.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += \
        gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)

    print 'adding a spill'
    spill = gnome.spill.PointLineSource(num_elements=1000,
            start_position=(-72.419992, 41.202120, 0.0),
            release_time=start_time)

    model.spills += spill

    print 'adding a RandomMover:'
    r_mover = gnome.movers.RandomMover(diffusion_coef=500000)
    model.movers += r_mover

    print 'adding a wind mover:'
    series = np.zeros((5, ), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (10, 45))
    series[1] = (start_time + timedelta(hours=18), (10, 90))
    series[2] = (start_time + timedelta(hours=30), (10, 135))
    series[3] = (start_time + timedelta(hours=42), (10, 180))
    series[4] = (start_time + timedelta(hours=54), (10, 225))

    wind = Wind(timeseries=series, units='m/s')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover

    print 'adding a cats mover:'
    curr_file = get_datafile(os.path.join(base_dir, r"./LI_tidesWAC.CUR"
                             ))
    tide_file = get_datafile(os.path.join(base_dir, r"./CLISShio.txt"))
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(tide_file))
    model.movers += c_mover
    model.environment += c_mover.tide

    print 'viewport is:', renderer.viewport

    return model


def post_run(model):

    # create a place for test images (cleaning out any old ones)

    images_dir = os.path.join(base_dir, 'images_2')
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    for outputter in model.outputters:
        if isinstance(outputter, gnome.renderer.Renderer):
            renderer = model.outputters[outputter.id]
            break

    renderer.images_dir = images_dir

    print 're-rendering images'

    # re-render images:

    renderer.viewport = ((-72.75, 41.1), (-72.34, 41.3))

    renderer.prepare_for_model_run(model.start_time)

    for step_num in range(model.num_time_steps):
        print 'writing image:'
        image_info = renderer.write_output(step_num)
        print 'image written:', image_info

    print 'viewport is:', renderer.viewport


if __name__ == '__main__':
    from gnome import scripting

    scripting.make_images_dir()
    model = make_model()
    model.full_run(log=True)
    post_run(model)

