#!/usr/bin/env python

"""
script that demonstrates outputting the surface_concentration
"""


#Really simple model!



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

and netcdf and kml output
"""


import os
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.projections import GeoProjection
from gnome.utilities.remote_data import get_datafile

from gnome.environment import Wind, Tide
from gnome.map import MapFromBNA

from gnome.model import Model
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover, CatsMover, ComponentMover


from gnome.outputters import Renderer, NetCDFOutput, KMZOutput, ShapeOutput

# let's get the console log working:
gnome.initialize_console_log()

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    # create the maps:

    start_time = datetime(2013, 3, 12, 10, 0)

    # 15 minutes in seconds
    # Default to now, rounded to the nearest hour
    model = Model(time_step=60 * 60,
                  start_time=start_time,
                  duration=timedelta(days=1),
                  uncertain=False)

    print 'adding outputters'
    renderer = Renderer(output_dir=images_dir)
                        # size=(800, 800),
                        # viewport=((-70.5, 41.5),
                                  # (-69.5, 42.5)),
                        # projection_class=GeoProjection)

    model.outputters += renderer
    # netcdf_file = os.path.join(base_dir, 'surface_concentration.nc')
    # scripting.remove_netcdf(netcdf_file)
    # model.outputters += NetCDFOutput(netcdf_file, surface_conc='kde')
    
    shape_file = os.path.join(base_dir, 'surface_concentration')
    model.outputters += ShapeOutput(shape_file, surface_conc='kde')

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=100000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 270))
    series[1] = (start_time + timedelta(hours=25), (5, 270))

    w_mover = WindMover(Wind(timeseries=series, units='m/s'))
    model.movers += w_mover
    model.environment += w_mover.wind

    print 'adding a spill'

    end_time = start_time + timedelta(hours=12)
    spill = point_line_release_spill(num_elements=100,
                                     start_position=(-70.0, 42, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time,
                                     amount=10000,
                                     units="barrels")

    model.spills += spill

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    print "setting up the model"
    model = make_model()
    print "running the model"
    for step in model:
        print step


