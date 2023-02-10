#!/usr/bin/env python
"""
a simple script to run GNOME

This one uses:

  - the GeoProjection
  - wind mover
  - random mover
  - cats shio mover
"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.projections import GeoProjection
from gnome.utilities.remote_data import get_datafile

from gnome.environment import Wind, Tide
from gnome.maps import MapFromBNA

from gnome.model import Model
from gnome.spills import surface_point_line_spill
from gnome.movers import RandomMover, PointWindMover, CatsMover


from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    print('creating the maps')
    mapfile = get_datafile(os.path.join(base_dir, 'SanJuanMap.bna'))
    gnome_map = MapFromBNA(mapfile, refloat_halflife=1,
                           raster_size=1024 * 1024)

    renderer = Renderer(mapfile,
                        images_dir,
                        image_size=(800, 800),
                        projection_class=GeoProjection)

    renderer.viewport = ((-66.24, 18.39), (-66.1, 18.55))

    print('initializing the model')
    start_time = datetime(2014, 9, 3, 13, 0)

    # 15 minutes in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=900, start_time=start_time,
                  duration=timedelta(days=1),
                  map=gnome_map, uncertain=False)

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_san_juan.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=100000)

    print('adding a wind mover:')

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (0, 270))
    series[1] = (start_time + timedelta(hours=18), (0, 270))

    wind = Wind(timeseries=series, units='m/s')
    w_mover = PointWindMover(wind)
    wind.extrapolation_is_allowed=True
    model.movers += w_mover

    print('adding a cats shio mover:')

    # need to add the scale_factor for the tide heights file
    curr_file = get_datafile(os.path.join(base_dir, 'EbbTides.cur'))
    tide_file = get_datafile(os.path.join(base_dir, 'EbbTidesShioHt.txt'))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file, scale_factor=.15))

    # this is the value in the file (default)
    c_mover.scale_refpoint = (-66.116667, 18.458333)
    c_mover.scale = True
    c_mover.scale_value = 1.0
    # c_mover.tide.scale_factor = 0.15

    model.movers += c_mover

    print('adding a cats mover:')

    curr_file = get_datafile(os.path.join(base_dir, 'Offshore.cur'))

    c_mover = CatsMover(curr_file)

    # this is the value in the file (default)
    # c_mover.scale_refpoint = (-66.082836, 18.469334)
    c_mover.scale_refpoint = (-66.084333333, 18.46966667)
    c_mover.scale = True
    c_mover.scale_value = 0.1

    model.movers += c_mover

    print('adding a spill')

    end_time = start_time + timedelta(hours=12)
    spill = surface_point_line_spill(num_elements=1000,
                                     release_time=start_time,
                                     start_position=(-66.16374,
                                                     18.468054, 0.0),
                                     # start_position=(-66.129099,
                                     #                 18.465332, 0.0),
                                     # end_release_time=end_time,
                                     )

    model.spills += spill

    return model

if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    model.full_run()
