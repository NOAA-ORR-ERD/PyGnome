#!/usr/bin/env python
"""
Script to test GNOME with plume element type
 - weibull droplet size distribution

Simple map and simple current mover

Rise velocity and vertical diffusion

This is simply making a point source with a given distribution of droplet sizes

"""

import os
from datetime import datetime, timedelta

import gnome.scripting as gs
from gnome.utilities.distributions import WeibullDistribution

from gnome.spills.substance import NonWeatheringSubstance
# from gnome.spills.initializers import plume_initializers

# define base directory
base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, 'images')


def make_model():
    print('initializing the model')

    start_time = datetime(2004, 12, 31, 13, 0)
    model = gs.Model(start_time=start_time,
                     duration=timedelta(days=3),
                     time_step=30 * 60,
                     uncertain=False)

    print('adding the map')
    model.map = gs.GnomeMap()

    renderer = gs.Renderer(output_dir=images_dir,
                           image_size=(800, 600),
                           output_timestep=timedelta(hours=1),
                           formats=['gif', 'png'],
                           point_size=4,
                           depth_colors='turbo',
                           min_color_depth=0,
                           max_color_depth=1700,
                           viewport = ((-77.0, 36.75), (-76.0, 37.75)),
                           )

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_plume.nc')
    gs.remove_netcdf(netcdf_file)

    model.outputters += gs.NetCDFOutput(netcdf_file, which_data='most',
                                        output_timestep=timedelta(hours=2))

    print('adding a spill')
    # Break the spill into two spills, first with the larger droplets
    # and second with the smaller droplets.
    # Split the total spill volume (100 m^3) to have most
    # in the larger droplet spill.
    # Smaller droplets start at a lower depth than larger

    wd = WeibullDistribution(alpha=1.8,
                             lambda_=.00456,
                             min_=.0002)  # 200 micron min
    end_time = start_time + timedelta(hours=24)

    spill = gs.subsurface_spill(num_elements=200,
                                start_position=(-76.126872, 37.680952, 1700.0),
                                release_time=start_time,
                                distribution=wd,
                                amount=90,  # default volume_units=m^3
                                units='m^3',
                                end_release_time=end_time,
                                substance=NonWeatheringSubstance(standard_density=900),
                                )

    model.spills += spill

    # wd = WeibullDistribution(alpha=1.8,
    #                          lambda_=.00456,
    #                          max_=.0002)  # 200 micron max

    # spill = gs.subsurface_spill(num_elements=50,
    #                             units='m^3',
    #                             start_position=(-76.126872, 37.680952, 1800.0),
    #                             release_time=start_time,
    #                             distribution = wd,
    #                             amount=90,
    #                             substance = NonWeatheringSubstance(standard_density=900),
    #                             )
    # model.spills += spill

    # print('adding a RandomMover:')
    # model.movers += gs.RandomMover(diffusion_coef=50000)

    print('adding a RiseVelocityMover:')
    model.movers += gs.RiseVelocityMover()

    print('adding a RandomMover3D:')
    model.movers += gs.RandomMover3D(vertical_diffusion_coef_above_ml=5,
                                     vertical_diffusion_coef_below_ml=.11,
                                     horizontal_diffusion_coef_above_ml=50000,
                                     horizontal_diffusion_coef_below_ml=5000,
                                     mixed_layer_depth=10)

    print('adding a wind mover:')
    model.movers += gs.constant_point_wind_mover(speed=30, direction=90, units='knot')

    print('adding a steady uniform current:')
    curr = gs.constant_point_current_mover(speed=.3, direction=180)
    model.movers += curr

    return model


if __name__ == "__main__":
    gs.make_images_dir(images_dir)
    model = make_model()
    print("about to start running the model")
    for step in model:
        print(f"step: {step['step_num']}")
        #print(step)
