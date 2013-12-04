#!/usr/bin/env python

"""
Script to test GNOME with san francisco bay data (NWS wind data)

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
    print 'initializing the model'

    start_time = datetime(2006, 3, 31, 21, 0)
    model = gnome.model.Model(start_time=start_time,
                              duration=timedelta(days=3), time_step=30
                              * 60, uncertain=True)  # 1 day of data in file
                                                      # 1/2 hr in seconds

    mapfile = get_datafile(os.path.join(base_dir, './coastSF.bna'
                           ))
    print 'adding the map'
    model.map = gnome.map.MapFromBNA(mapfile, refloat_halflife=1)  # seconds

    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800,
            600),draw_ontop='forecast')
    renderer.viewport = ((-124.5, 37.), (-120.5, 39))

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_sf_bay.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += \
        gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)

    print 'adding a spill'

    spill = gnome.spill.PointLineSource(num_elements=1000,
            start_position=(-123.57152, 37.369436, 0.0),
            release_time=start_time,
            windage_range=(.01,.04)
            )  

    model.spills += spill

#     print 'adding a RandomMover:'
#     r_mover = gnome.movers.RandomMover(diffusion_coef=50000)
#     model.movers += r_mover
# 
    print 'adding a grid wind mover:'

    wind_file = get_datafile(os.path.join(base_dir,
                             r"./WindSpeedDirSubset.nc"))
    topology_file = get_datafile(os.path.join(base_dir,
                                 r"./WindSpeedDirSubsetTop.dat"))
    w_mover = gnome.movers.GridWindMover(wind_file, topology_file)
    #w_mover.uncertain_time_delay=6
    #w_mover.uncertain_duration=6
    w_mover.uncertain_speed_scale=1
    w_mover.set_uncertain_angle(.2,'rad')	#default is .4
    w_mover.wind_scale=2
    model.movers += w_mover

    return model


