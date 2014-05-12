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

import numpy
np = numpy

from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.projections import GeoProjection
from gnome.utilities.remote_data import get_datafile

from gnome.environment import Wind, Tide
from gnome.map import MapFromBNA

from gnome.model import Model
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover, CatsMover, ComponentMover


from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    print 'creating the maps'
    mapfile = get_datafile(os.path.join(base_dir, './DelawareRiverMap.bna'))
    gnome_map = MapFromBNA(mapfile, refloat_halflife=1)  # hours

    renderer = Renderer(mapfile, images_dir, size=(800, 800),
                        projection_class=GeoProjection)

    print 'initializing the model'
    start_time = datetime(2012, 8, 20, 13, 0)

    # 15 minutes in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=900, start_time=start_time,
                  duration=timedelta(days=1),
                  map=gnome_map, uncertain=False)

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_delaware_bay.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=100000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 270))
    series[1] = (start_time + timedelta(hours=18), (5, 270))

    wind_file = get_datafile(os.path.join(base_dir, r"ConstantWind.WND"))
   # wind = Wind(filename=wind_file)
    wind = Wind(timeseries=series, units='m/s')
    w_mover = WindMover(wind)
    #w_mover = WindMover(Wind(timeseries=series, units='knots'))
    model.movers += w_mover

    print 'adding a cats shio mover:'

    curr_file = get_datafile(os.path.join(base_dir, r"./FloodTides.cur"))
    tide_file = get_datafile(os.path.join(base_dir, r"./FloodTidesShio.txt"))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # this is the value in the file (default)
    c_mover.scale_refpoint = (-75.081667,38.7995)
    c_mover.scale = True
    c_mover.scale_value = 1

    model.movers += c_mover

    # TODO: cannot add this till environment base class is created
    model.environment += c_mover.tide

    print 'adding a cats mover:'

    curr_file = get_datafile(os.path.join(base_dir,
                             r"./Offshore.cur"))
    c_mover = CatsMover(curr_file)

    # but do need to scale (based on river stage)

    c_mover.scale = True
    c_mover.scale_refpoint = (-74.7483333,38.898333)
    c_mover.scale_value = .03
    model.movers += c_mover
# 
# pat1Angle 315; pat1Speed 30; pat1SpeedUnits knots; pat1ScaleToValue 0.314426 # these are from windows they don't match Mac values...
# pat2Angle 225; pat2Speed 30; pat2SpeedUnits knots; pat2ScaleToValue 0.032882
# scaleBy WindStress
 
    print 'adding a component mover:'

    curr_file1 = get_datafile(os.path.join(base_dir, r"NW30ktwinds.cur"))
    curr_file2 = get_datafile(os.path.join(base_dir, r"SW30ktwinds.cur"))
    comp_mover = ComponentMover(curr_file1, curr_file2, wind)
    #todo: following is not working when model is saved out - fix
    #comp_mover = ComponentMover(curr_file1, curr_file2, Wind(timeseries=series, units='m/s'))
    #comp_mover = ComponentMover(curr_file1, curr_file2, wind=Wind(filename=wind_file))

    comp_mover.ref_point = (-75.263166,39.1428333)
    comp_mover.pat1_angle = 315
    comp_mover.pat1_speed = 30
    comp_mover.pat1_speed_units = 1
    #comp_mover.pat1ScaleToValue = .314426
    comp_mover.pat1_scale_to_value = .502035
    comp_mover.pat2_angle = 225
    comp_mover.pat2_speed = 30
    comp_mover.pat2_speed_units = 1
    #comp_mover.pat2ScaleToValue = .032882
    comp_mover.pat2_scale_to_value = .021869

    model.movers += comp_mover

    print 'adding a spill'

    end_time = start_time + timedelta(hours=12)
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-75.262319,
                                                     39.142987, 0.0),
                                     release_time=start_time)
                                     #end_release_time=end_time)

    model.spills += spill

    return model
