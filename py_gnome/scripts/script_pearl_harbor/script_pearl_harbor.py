#!/usr/bin/env python
"""
Script to test GNOME with CH3D data for Pearl Harbor.

Returns an empty model if it cannot download the Pearl Harbor maps or datafiles
"""

import os
from datetime import datetime, timedelta
from urllib2 import HTTPError

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

    start_time = datetime(2013, 1, 1, 1)  # data starts at 1:00 instead of 0:00
    model = Model(start_time=start_time, duration=timedelta(days=1),
                  time_step=900, uncertain=False)

    try:
        mapfile = get_datafile(os.path.join(base_dir, './pearl_harbor.bna'))
    except HTTPError:
        print ('Could not download Pearl Harbor data from server - '
               'returning empty model')
        return model

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=1)  # hours

    #
    # Add the outputters -- render to images, and save out as netCDF
    #

    print 'adding renderer and netcdf output'
    model.outputters += Renderer(mapfile, images_dir, size=(800, 600))

    netcdf_output_file = os.path.join(base_dir, 'pearl_harbor_output.nc')
    scripting.remove_netcdf(netcdf_output_file)
    model.outputters += NetCDFOutput(netcdf_output_file, which_data='all')

    # #
    # # Set up the movers:
    # #
    print 'adding a random mover:'
    model.movers += RandomMover(diffusion_coef=10000)

    print 'adding a wind mover:'
    series = np.zeros((3, ), dtype=datetime_value_2d)
    series[0] = (start_time, (4, 180))
    series[1] = (start_time + timedelta(hours=12), (2, 270))
    series[2] = (start_time + timedelta(hours=24), (4, 180))

    w_mover = WindMover(Wind(timeseries=series, units='knots'))
    model.movers += w_mover
    model.environment += w_mover.wind

    print 'adding a current mover:'
    # this is CH3D currents
    curr_file = os.path.join(base_dir, r"./ch3d2013.nc")
    topology_file = os.path.join(base_dir, r"./PearlHarborTop.dat")
    model.movers += GridCurrentMover(curr_file, topology_file)

    # #
    # # Add a spill (sources of elements)
    # #
    print 'adding spill'
    model.spills += point_line_release_spill(num_elements=1000,
                                             start_position=(-157.97064,
                                                             21.331524, 0.0),
                                             release_time=start_time)
    return model


if __name__ == '__main__':
    scripting.make_images_dir()
    model = make_model()
    model.full_run(log=True)
