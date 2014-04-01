#!/usr/bin/env python
"""
Script to test GNOME with HYCOM data in Mariana Islands region.
"""

NUM_ELEMENTS = 1e6

import os
import shutil
from datetime import datetime, timedelta

import numpy
np = numpy

import gnome
from gnome import scripting
from gnome import utilities
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

    start_time = datetime(2013, 5, 18, 0)

    model = Model(start_time=start_time, duration=timedelta(days=8),
                  time_step=1 * 3600, uncertain=False)

    mapfile = get_datafile(os.path.join(base_dir, './mariana_island.bna'))

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=6)  # hours

    #
    # Add the outputters -- render to images, and save out as netCDF
    #

    print 'adding renderer'
    model.outputters += Renderer(mapfile, images_dir, size=(800, 600))

    # print "adding netcdf output"
    # netcdf_output_file = os.path.join(base_dir,'mariana_output.nc')
    # scripting.remove_netcdf(netcdf_output_file)
    # model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_output_file,
    #                                                         which_data='all')

    # #
    # # Set up the movers:
    # #

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=10000)

    print 'adding a simple wind mover:'
    model.movers += constant_wind_mover(5, 315, units='m/s')

    print 'adding a current mover:'

    # # this is HYCOM currents
    curr_file = get_datafile(os.path.join(base_dir, r"./HYCOM.nc"))
    model.movers += GridCurrentMover(curr_file)

    # #
    # # Add some spills (sources of elements)
    # #

    print 'adding four spill'
    model.spills += point_line_release_spill(num_elements=NUM_ELEMENTS // 4,
                                             start_position=(145.25, 15.0,
                                                             0.0),
                                             release_time=start_time)
    model.spills += point_line_release_spill(num_elements=NUM_ELEMENTS // 4,
                                             start_position=(146.25, 15.0,
                                                             0.0),
                                             release_time=start_time)
    model.spills += point_line_release_spill(num_elements=NUM_ELEMENTS // 4,
                                             start_position=(145.75, 15.25,
                                                             0.0),
                                             release_time=start_time)
    model.spills += point_line_release_spill(num_elements=NUM_ELEMENTS // 4,
                                             start_position=(145.75, 14.75,
                                                             0.0),
                                             release_time=start_time)

    return model


if __name__ == '__main__':
    scripting.make_images_dir()
    model = make_model()
    for step in model:
        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    #model.full_run(log=True)
