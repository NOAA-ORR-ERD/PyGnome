#!/usr/bin/env python
"""
Script to test GNOME with long island sound data
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
from gnome.environment import Wind, Tide
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover, CatsMover

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

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
                  map=gnome_map, uncertain=True, cache_enabled=True)

    print 'adding outputters'
    model.outputters += Renderer(mapfile, images_dir, size=(800, 600))

    netcdf_file = os.path.join(base_dir, 'script_long_island.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print 'adding a spill'
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-72.419992,
                                                     41.202120, 0.0),
                                     release_time=start_time)
    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=500000,uncertain_factor=2)

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

    print 'viewport is:', [o.viewport
                           for o in model.outputters
                           if isinstance(o, Renderer)]

    return model


def post_run(model):

    # create a place for test images (cleaning out any old ones)
    images_dir = os.path.join(base_dir, 'images_2')
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)

    os.mkdir(images_dir)

    renderers = [o for o in model.outputters
                 if isinstance(o, Renderer)]

    print 're-rendering images'
    if renderers:
        renderer = renderers[0]

        renderer.images_dir = images_dir
        renderer.viewport = ((-72.75, 41.1), (-72.34, 41.3))

        renderer.prepare_for_model_run(model.start_time)

        for step_num in range(model.num_time_steps):
            print 'writing image:'
            image_info = renderer.write_output(step_num)
            print 'image written:', image_info

        print 'viewport is:', renderer.viewport
    else:
        print 'No Renderers available!!!'


if __name__ == '__main__':
    scripting.make_images_dir()

    model = make_model()

    model.full_run(log=True)
    post_run(model)
