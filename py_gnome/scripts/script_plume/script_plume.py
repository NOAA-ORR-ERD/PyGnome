#!/usr/bin/env python
"""
Script to test GNOME with plume element type
 - weibull droplet size distribution
Simple map and simple current mover
Rise velocity and vertical diffusion
"""

import os
from datetime import datetime, timedelta

import numpy
np = numpy

from gnome import scripting
from gnome.spill.elements import plume
from gnome.utilities.distributions import WeibullDistribution

from gnome.model import Model
from gnome.map import GnomeMap
from gnome.spill import point_line_release_spill
from gnome.movers import (RandomMover,
                          RiseVelocityMover,
                          RandomVerticalMover,
                          SimpleMover)

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2004, 12, 31, 13, 0)
    model = Model(start_time=start_time, duration=timedelta(days=3),
                  time_step=30 * 60, uncertain=False)

    print 'adding the map'
    model.map = GnomeMap()

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(images_dir=images_dir,
                        #size=(800, 600),
                        output_timestep=timedelta(hours=1),
                        draw_ontop='uncertain')
    renderer.viewport = ((-76.5, 37.), (-75.8, 38.))

    print 'adding outputters'
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_plume.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file, which_data='most',
                                     output_timestep=timedelta(hours=2))

    print 'adding two spills'
    # Break the spill into two spills, first with the larger droplets
    # and second with the smaller droplets.
    # Split the total spill volume (100 m^3) to have most
    # in the larger droplet spill.
    # Smaller droplets start at a lower depth than larger

    wd = WeibullDistribution(alpha=1.8, lambda_=.00456,
                             min_=.0002)  # 200 micron min
    end_time = start_time + timedelta(hours=24)
    spill = point_line_release_spill(num_elements=1000,
                                     volume=90,  # default volume_units=m^3
                                     start_position=(-76.126872, 37.680952,
                                                     1700),
                                     release_time=start_time,
                                     end_release_time=end_time,
                                     element_type=plume(distribution=wd))
    model.spills += spill

    wd = WeibullDistribution(alpha=1.8, lambda_=.00456,
                             max_=.0002)  # 200 micron max
    spill = point_line_release_spill(num_elements=1000, volume=10,
                                     start_position=(-76.126872, 37.680952,
                                                     1800),
                                     release_time=start_time,
                                     element_type=plume(distribution=wd))
    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=50000)

    print 'adding a RiseVelocityMover:'
    model.movers += RiseVelocityMover()

    print 'adding a RandomVerticalMover:'
    model.movers += RandomVerticalMover(vertical_diffusion_coef_above_ml=5,
                                        vertical_diffusion_coef_below_ml=.11,
                                        mixed_layer_depth=10)

    # print 'adding a wind mover:'

    # series = np.zeros((2, ), dtype=gnome.basic_types.datetime_value_2d)
    # series[0] = (start_time, (30, 90))
    # series[1] = (start_time + timedelta(hours=23), (30, 90))

    # wind = Wind(timeseries=series, units='knot')
    #
    # default is .4 radians
    # w_mover = gnome.movers.WindMover(wind, uncertain_angle_scale=0)
    #
    # model.movers += w_mover

    print 'adding a simple mover:'
    s_mover = SimpleMover(velocity=(0.0, -.1, 0.0))
    model.movers += s_mover

    return model


if __name__ == "__main__":
    model = make_model()
    scripting.make_images_dir()
    model = make_model()
    for step in model:
        print step
