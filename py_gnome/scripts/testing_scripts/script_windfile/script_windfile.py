#!/usr/bin/env python

"""
Script to test "bad" nws data file

Data file downloaded via GOODS on 3/9/2016 -- that same point source fails
when used thorugh WebGNOME

"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.remote_data import get_datafile

from gnome.environment import Wind
# from gnome.environment import Tide
from gnome.maps import MapFromBNA

from gnome.model import Model
from gnome.spills import surface_point_line_spill
from gnome.movers import RandomMover, PointWindMover
# from gnome.movers import CatsMover, ComponentMover


from gnome.outputters import Renderer
# from gnome.outputters import  NetCDFOutput, KMZOutput

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:
    print('creating the maps')
    mapfile = get_datafile(os.path.join(base_dir, './MassBayMap.bna'))
    gnome_map = MapFromBNA(mapfile,
                           refloat_halflife=1,  # hours
                           raster_size=2048*2048  # about 4 MB
                           )

    renderer = Renderer(mapfile,
                        images_dir,
                        image_size=(800, 800),
                        )

    print('initializing the model')
    start_time = datetime(2016, 3, 9, 15)

    # 1 hour in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=3600,
                  start_time=start_time,
                  duration=timedelta(days=6),
                  map=gnome_map,
                  uncertain=True)

    print('adding outputters')
    model.outputters += renderer

    # netcdf_file = os.path.join(base_dir, 'script_boston.nc')
    # scripting.remove_netcdf(netcdf_file)
    # model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    # model.outputters += KMZOutput(os.path.join(base_dir, 'script_boston.kmz'))

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=100000)

    print('adding a wind mover:')

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + timedelta(hours=18), (5, 180))
    w = Wind(filename=os.path.join(base_dir, '22NM_WNW_PortAngelesWA.nws'))
    w_mover = PointWindMover(w)
    model.movers += w_mover
    model.environment += w_mover.wind

    # print 'adding a cats shio mover:'

    # curr_file = get_datafile(os.path.join(base_dir, r"./EbbTides.cur"))
    # tide_file = get_datafile(os.path.join(base_dir, r"./EbbTidesShio.txt"))

    # c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # # this is the value in the file (default)
    # c_mover.scale_refpoint = (-70.8875, 42.321333)
    # c_mover.scale = True
    # c_mover.scale_value = -1

    # model.movers += c_mover

    # # TODO: cannot add this till environment base class is created
    # model.environment += c_mover.tide

    print('adding a spill')

    end_time = start_time + timedelta(hours=12)
    spill = surface_point_line_spill(num_elements=100,
                                     start_position=(-70.911432,
                                                     42.369142, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time)

    model.spills += spill

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    print("setting up the model")
    model = make_model()
    print("running the model")
    model.full_run()