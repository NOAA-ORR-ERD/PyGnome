#!/usr/bin/env python

"""
a simple script to run GNOME
"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind
from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome.utilities.remote_data import get_datafile
from gnome import scripting

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    print 'creating the maps'

    mapfile = get_datafile(os.path.join(base_dir,
                           './LowerMississippiMap.bna'))
    gnome_map = gnome.map.MapFromBNA(mapfile, refloat_halflife=6)  # hours

    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800,
            600))

    print 'initializing the model'

    start_time = datetime(2012, 9, 15, 12, 0)
    model = gnome.model.Model(time_step=600, start_time=start_time,
                              duration=timedelta(days=1),
                              map=gnome_map, uncertain=True)  # 10 minutes in seconds
                                                              # default to now, rounded to the nearest hour

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_lower_mississippi.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += \
        gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)

    print 'adding a RandomMover:'
    model.movers += gnome.movers.RandomMover(diffusion_coef=10000)

    print 'adding a wind mover:'

    series = np.zeros((5, ), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (2, 45))
    series[1] = (start_time + timedelta(hours=18), (2, 90))
    series[2] = (start_time + timedelta(hours=30), (2, 135))
    series[3] = (start_time + timedelta(hours=42), (2, 180))
    series[4] = (start_time + timedelta(hours=54), (2, 225))

    wind = Wind(timeseries=series, units='m/s')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover

    print 'adding a cats mover:'

    curr_file = get_datafile(os.path.join(base_dir, r"LMiss.CUR"))
    c_mover = gnome.movers.CatsMover(curr_file)

    # but do need to scale (based on river stage)

    c_mover.scale = True
    c_mover.scale_refpoint = (-89.699944, 29.494558)
    c_mover.scale_value = 1.027154  # based on stage height 10ft (range is 0-18)
    model.movers += c_mover

    print 'adding a spill'

    spill = gnome.spill.PointSourceSurfaceRelease(num_elements=1000,
            start_position=(-89.699944, 29.494558, 0.0),
            release_time=start_time)

    model.spills += spill

    return model


