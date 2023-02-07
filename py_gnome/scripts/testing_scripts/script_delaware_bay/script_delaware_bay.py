#!/usr/bin/env python
"""
a simple script to run GNOME

This one uses:

  - the GeoProjection
  - wind mover
  - random mover
  - cats shio mover
  - plain cats mover
  - component mover
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
from gnome.movers import RandomMover, PointWindMover, CatsMover, ComponentMover


from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    print('creating the maps')
    mapfile = get_datafile(os.path.join(base_dir, 'DelawareRiverMap.bna'))
    gnome_map = MapFromBNA(mapfile, refloat_halflife=1)  # hours

    renderer = Renderer(mapfile, images_dir, image_size=(800, 800),
                        projection_class=GeoProjection)

    print('initializing the model')
    start_time = datetime(2012, 8, 20, 13, 0)

    # 15 minutes in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=900, start_time=start_time,
                  duration=timedelta(days=1),
                  map=gnome_map, uncertain=False)

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_delaware_bay.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=100000)

    print('adding a wind mover:')

    # wind_file = get_datafile(os.path.join(base_dir, 'ConstantWind.WND'))
    # wind = Wind(filename=wind_file)

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 270))
    series[1] = (start_time + timedelta(hours=25), (5, 270))

    wind = Wind(timeseries=series, units='m/s')

    # w_mover = PointWindMover(Wind(timeseries=series, units='knots'))
    w_mover = PointWindMover(wind)
    model.movers += w_mover

    print('adding a cats shio mover:')

    curr_file = get_datafile(os.path.join(base_dir, 'FloodTides.cur'))
    tide_file = get_datafile(os.path.join(base_dir, 'FloodTidesShio.txt'))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # this is the value in the file (default)
    c_mover.scale_refpoint = (-75.081667, 38.7995)
    c_mover.scale = True
    c_mover.scale_value = 1

    model.movers += c_mover

    # TODO: cannot add this till environment base class is created
    model.environment += c_mover.tide

    print('adding a cats mover:')

    curr_file = get_datafile(os.path.join(base_dir, 'Offshore.cur'))
    c_mover = CatsMover(curr_file)

    # but do need to scale (based on river stage)

    c_mover.scale = True
    c_mover.scale_refpoint = (-74.7483333, 38.898333)
    c_mover.scale_value = .03
    model.movers += c_mover
    #
    # these are from windows they don't match Mac values...
    # pat1Angle 315;
    # pat1Speed 30; pat1SpeedUnits knots;
    # pat1ScaleToValue 0.314426
    #
    # pat2Angle 225;
    # pat2Speed 30; pat2SpeedUnits knots;
    # pat2ScaleToValue 0.032882
    # scaleBy WindStress

    print('adding a component mover:')

    # if only using one current pattern
    # comp_mover = ComponentMover(curr_file1, None, wind)
    #
    # todo: following is not working when model is saved out - fix
    # comp_mover = ComponentMover(curr_file1, curr_file2,
    #                             Wind(timeseries=series, units='m/s'))
    # comp_mover = ComponentMover(curr_file1, curr_file2,
    #                             wind=Wind(filename=wind_file))

    curr_file1 = get_datafile(os.path.join(base_dir, 'NW30ktwinds.cur'))
    curr_file2 = get_datafile(os.path.join(base_dir, 'SW30ktwinds.cur'))
    comp_mover = ComponentMover(curr_file1, curr_file2, wind)

    comp_mover.scale_refpoint = (-75.263166, 39.1428333)

    comp_mover.pat1_angle = 315
    comp_mover.pat1_speed = 30
    comp_mover.pat1_speed_units = 1
    # comp_mover.pat1ScaleToValue = .314426
    comp_mover.pat1_scale_to_value = .502035

    comp_mover.pat2_angle = 225
    comp_mover.pat2_speed = 30
    comp_mover.pat2_speed_units = 1
    # comp_mover.pat2ScaleToValue = .032882
    comp_mover.pat2_scale_to_value = .021869

    model.movers += comp_mover

    print('adding a spill')

    end_time = start_time + timedelta(hours=12)
    spill = surface_point_line_spill(num_elements=1000,
                                     release_time=start_time,
                                     # end_release_time=end_time,
                                     start_position=(-75.262319,
                                                     39.142987, 0.0),
                                     )

    model.spills += spill

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    model.full_run()
