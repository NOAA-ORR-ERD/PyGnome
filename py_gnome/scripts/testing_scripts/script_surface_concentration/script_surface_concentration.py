#!/usr/bin/env python

"""
script that demonstrates outputting the surface_concentration
"""

# Really simple model!


import os
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome import scripting
from gnome.basic_types import datetime_value_2d

from gnome.utilities.projections import GeoProjection

from gnome.environment import Wind

from gnome.model import Model
from gnome.spills import surface_point_line_spill

from gnome.movers import RandomMover, PointWindMover

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

    print('adding outputters')
    renderer = Renderer(output_dir=images_dir,
                        image_size=(800, 800),
                        # viewport=((-70.25, 41.75), # FIXME -- why doesn't this work?
                        #           (-69.75, 42.25)),
                        projection_class=GeoProjection)
    renderer.viewport = ((-70.25, 41.75),
                         (-69.75, 42.25))
    model.outputters += renderer
    netcdf_file = os.path.join(base_dir, 'surface_concentration.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, surface_conc='kde')

    shape_file = os.path.join(base_dir, 'surface_concentration')
    model.outputters += ShapeOutput(shape_file, surface_conc='kde')

    shp_file = os.path.join(base_dir, 'surface_concentration')
    scripting.remove_netcdf(shp_file + ".zip")
    model.outputters += ShapeOutput(shp_file,
                                    zip_output=False,
                                    surface_conc="kde",
                                    )

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=100000)

    print('adding a wind mover:')

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 270))
    series[1] = (start_time + timedelta(hours=25), (5, 270))

    w_mover = PointWindMover(Wind(timeseries=series, units='m/s'))
    model.movers += w_mover
    model.environment += w_mover.wind

    print('adding a spill')

    end_time = start_time + timedelta(hours=12)
    spill = surface_point_line_spill(num_elements=100,
                                     amount=10000,
                                     units='gal',
                                     start_position=(-70.0, 42, 0.0),
                                     release_time=start_time,
                                     end_release_time=end_time,
                                     )

    model.spills += spill

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    print("setting up the model")
    model = make_model()
    print("running the model")
    for step in model:
        print(step)


