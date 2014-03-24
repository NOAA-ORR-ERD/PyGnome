#!/usr/bin/env python
"""
Script to test GNOME with guam data
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
from gnome.environment import Wind, Tide
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover, CatsMover

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2013, 2, 13, 9, 0)

    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(days=2),
                  time_step=30 * 60,
                  uncertain=False)

    print 'adding the map'
    mapfile = get_datafile(os.path.join(base_dir, './GuamMap.bna'))
    model.map = MapFromBNA(mapfile, refloat_halflife=6)  # hours

    print 'adding outputters'
    renderer = Renderer(mapfile, images_dir, size=(800, 600))
    renderer.viewport = ((144.6, 13.4), (144.7, 13.5))
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_guam.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print 'adding a spill'
    end_time = start_time + timedelta(hours=6)
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(144.664166,
                                                     13.441944, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time)
    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=50000)

    print 'adding a wind mover:'
    series = np.zeros((4, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 135))
    series[1] = (start_time + timedelta(hours=23), (5, 135))
    series[2] = (start_time + timedelta(hours=25), (5, 0))
    series[3] = (start_time + timedelta(hours=48), (5, 0))

    wind = Wind(timeseries=series, units='knot')
    w_mover = WindMover(wind)
    model.movers += w_mover
    model.environment += w_mover.wind

    print 'adding a cats mover:'
    curr_file = get_datafile(os.path.join(base_dir, r"./OutsideWAC.cur"))
    c_mover = CatsMover(curr_file)

    c_mover.scale = True
    c_mover.scale_refpoint = (144.601, 13.42)
    c_mover.scale_value = .15

    model.movers += c_mover

    print 'adding a cats shio mover:'
    curr_file = get_datafile(os.path.join(base_dir, r"./WACFloodTide.cur"))
    tide_file = get_datafile(os.path.join(base_dir, r"./WACFTideShioHts.txt"))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # this is different from the value in the file!
    c_mover.scale_refpoint = (144.621667, 13.45)

    c_mover.scale = True
    c_mover.scale_value = 1

    # will need the fScaleFactor for heights files
    c_mover.tide.scale_factor = 1.1864

    # will need the fScaleFactor for heights files
    #c_mover.time_dep.scale_factor = 1.1864

    model.movers += c_mover
    model.environment += c_mover.tide

    return model
