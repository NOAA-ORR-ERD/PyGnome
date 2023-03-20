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
import gnome
import gnome.scripting as gs
from gnome.utilities.distributions import WeibullDistribution

from gnome.spills.substance import NonWeatheringSubstance
from gnome.environment.environment_objects import GridCurrent

# define base directory
base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, 'images')

def make_model():
    print('initializing the model')
    start_time = datetime(2022, 12, 14, 21, 0)
    model = gs.Model(start_time=start_time,
                     duration=timedelta(hours=48),
                     time_step= 15 * 60,
                     uncertain=False)

    print('adding the map')
    mapfile = gs.get_datafile(os.path.join(base_dir, 'coast_HI.bna'))
    model.map = gs.MapFromBNA(mapfile, refloat_halflife=0.0)

    print('adding outputters')
    renderer = gs.Renderer(output_dir=images_dir,
                           image_size=(800, 600),
                           output_timestep=timedelta(hours=1),
                           draw_ontop='uncertain')
    renderer.viewport = ((-155.5, 19), (-157.5, 21))
    model.outputters += renderer

    print('adding spills')
    wd = WeibullDistribution(alpha=1.8,
                             lambda_=.00456,
                             min_=.0002)  # 200 micron min

    spill = gs.subsurface_spill(num_elements=500,
                                start_position=(-156.59, 20.021, 60.0),
                                release_time=start_time,
                                distribution=wd,
                                amount=90,  # default volume_units=m^3
                                units='m^3',
                                end_release_time=start_time,
                                substance=NonWeatheringSubstance(standard_density=900),
                                )

    model.spills += spill

    print('adding a RiseVelocityMover:')
    model.movers += gs.RiseVelocityMover()

    print('adding a 3DCurrentMover')
    file = gs.get_datafile(os.path.join(base_dir, '3Dcurrents_HI.nc'))
    current = gs.GridCurrent.from_netCDF(filename=file)
    model.movers += gs.CurrentMover(current=current)

    print('adding a RandomMover3D:')
    model.movers += gs.RandomMover3D(vertical_diffusion_coef_above_ml=5,
                                     vertical_diffusion_coef_below_ml=.11,
                                     horizontal_diffusion_coef_above_ml = 500000,
                                     horizontal_diffusion_coef_below_ml = 100000,
                                     mixed_layer_depth=20)

    return model


if __name__ == "__main__":
    gs.make_images_dir(images_dir)
    model = make_model()
    print("about to start running the model")
    for step in model:
        print(f"step: {step['step_num']}")
        #print(step)
