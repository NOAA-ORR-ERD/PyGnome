#!/usr/bin/env python
"""
a simple script to run GNOME

This one uses:

  - the GeoProjection
  - wind mover
  - random mover
  - cats shio mover
  - cats ossm mover
  - plain cats mover
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
from gnome.movers import RandomMover, WindMover, CatsMover


from gnome.renderer import Renderer
from gnome.outputters import NetCDFOutput

# define base directory

base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    print 'creating the maps'
    mapfile = get_datafile(os.path.join(base_dir, './MassBayMap.bna'))
    gnome_map = MapFromBNA(mapfile, refloat_halflife=1)  # hours

    renderer = Renderer(mapfile, images_dir, size=(800, 800),
                        projection_class=GeoProjection)

    print 'initializing the model'
    start_time = datetime(2013, 3, 12, 10, 0)

    # 15 minutes in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=900, start_time=start_time,
                  duration=timedelta(days=1),
                  map=gnome_map, uncertain=False)

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_boston.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=100000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + timedelta(hours=18), (5, 180))

    w_mover = WindMover(Wind(timeseries=series, units='m/s'))
    model.movers += w_mover
    model.environment += w_mover.wind

    print 'adding a cats shio mover:'

    curr_file = get_datafile(os.path.join(base_dir, r"./EbbTides.cur"))
    tide_file = get_datafile(os.path.join(base_dir, r"./EbbTidesShio.txt"))

    c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # this is the value in the file (default)
    c_mover.scale_refpoint = (-70.8875, 42.321333)
    c_mover.scale = True
    c_mover.scale_value = -1

    model.movers += c_mover

    # TODO: cannot add this till environment base class is created
    model.environment += c_mover.tide

    print 'adding a cats ossm mover:'

    #ossm_file = get_datafile(os.path.join(base_dir,
    #                         r"./MerrimackMassCoastOSSM.txt"))
    curr_file = get_datafile(os.path.join(base_dir,
                             r"./MerrimackMassCoast.cur"))
    tide_file = get_datafile(os.path.join(base_dir,
                             r"./MerrimackMassCoastOSSM.txt"))
    c_mover = CatsMover(curr_file, tide=Tide(tide_file))

    # but do need to scale (based on river stage)

    c_mover.scale = True
    c_mover.scale_refpoint = (-70.65, 42.58333)
    c_mover.scale_value = 1.
    model.movers += c_mover
    model.environment += c_mover.tide

    print 'adding a cats mover:'

    curr_file = get_datafile(os.path.join(base_dir, r"MassBaySewage.cur"))
    c_mover = CatsMover(curr_file)

    # but do need to scale (based on river stage)

    c_mover.scale = True
    c_mover.scale_refpoint = (-70.78333, 42.39333)

    # the scale factor is 0 if user inputs no sewage outfall effects
    c_mover.scale_value = .04

    model.movers += c_mover

    # print "adding a component mover:"
    # component_file1 =  os.path.join( base_dir, r"./WAC10msNW.cur")
    # component_file2 =  os.path.join( base_dir, r"./WAC10msSW.cur")

    print 'adding a spill'

    end_time = start_time + timedelta(hours=12)
    spill = point_line_release_spill(num_elements=1000,
                                     start_position=(-70.911432,
                                                     42.369142, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time)

    model.spills += spill

    return model
