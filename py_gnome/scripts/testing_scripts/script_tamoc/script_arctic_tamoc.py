#!/usr/bin/env python
"""
Script to test GNOME with:

TAMOC - Texas A&M Oilspill Calculator

https://github.com/socolofs/tamoc

This is a very simpile environment:

Simple map (no land) and simple current mover (steady uniform current)

Rise velocity and vertical diffusion

But it's enough to see if the coupling with TAMOC works.

"""


import os
import numpy as np
from datetime import datetime, timedelta

from gnome import scripting
from gnome.utilities.distributions import WeibullDistribution
from gnome.environment.gridded_objects_base import Variable, Grid_S
from gnome.environment import IceAwareCurrent, IceConcentration, IceVelocity

from gnome.model import Model
from gnome.maps.map import GnomeMap
from gnome.movers import (RandomMover,
                          TamocRiseVelocityMover,
                          RandomMover3D,
                          SimpleMover,
                          c_GridCurrentMover,
                          CurrentMover,
                          constant_point_wind_mover,
                          IceMover)

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput
from gnome.tamoc import tamoc_spill
from gnome.environment.environment_objects import IceAwareCurrent

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    # set up the modeling environment
    start_time = datetime(2016, 9, 23, 0, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=2),
                  time_step=30 * 60,
                  uncertain=False)

    print('adding the map')
    model.map = GnomeMap()  # this is a "water world -- no land anywhere"

    # renderere is only top-down view on 2d -- but it's something
    renderer = Renderer(output_dir=images_dir,
                        image_size=(1024, 768),
                        output_timestep=timedelta(hours=1),
                        )
    renderer.viewport = ((196.14, 71.89), (196.18, 71.93))

    print('adding outputters')
    model.outputters += renderer

    # Also going to write the results out to a netcdf file
    netcdf_file = os.path.join(base_dir, 'script_arctic_plume.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file,
                                     which_data='most',
                                     # output most of the data associated with the elements
                                     output_timestep=timedelta(hours=2))

    print("adding Horizontal and Vertical diffusion")

    # Horizontal Diffusion
    model.movers += RandomMover(diffusion_coef=500)
    # vertical diffusion (different above and below the mixed layer)
    model.movers += RandomMover3D(vertical_diffusion_coef_above_ml=5,
                                        vertical_diffusion_coef_below_ml=.11,
                                        mixed_layer_depth=10)

    print('adding Rise Velocity')
    # droplets rise as a function of their density and radius
    model.movers += TamocRiseVelocityMover()

    print('adding a circular current and eastward current')
    fn = 'hycom_glb_regp17_2016092300_subset.nc'
    fn_ice = 'hycom-cice_ARCu0.08_046_2016092300_subset.nc'
    iconc = IceConcentration.from_netCDF(filename=fn_ice)
    ivel = IceVelocity.from_netCDF(filename=fn_ice, grid = iconc.grid)
    ic = IceAwareCurrent.from_netCDF(ice_concentration = iconc, ice_velocity= ivel, filename=fn)

    model.movers += CurrentMover(current = ic)
    model.movers += SimpleMover(velocity=(0., 0., 0.))
    model.movers += constant_point_wind_mover(20, 315, units='knots')

    # Now to add in the TAMOC "spill"
    print("Adding TAMOC spill")

    model.spills += tamoc_spill.TamocSpill(release_time=start_time,
                                        start_position=(196.16, 71.91, 40.0),
                                        num_elements=1000,
                                        #end_release_time=start_time + timedelta(days=1),
                                        release_duration=timedelta(days=1),
                                        name='TAMOC plume',
                                        #TAMOC_interval=None,  # how often to re-run TAMOC
                                        )

    #model.spills[0].data_sources['currents'] = ic

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    #model.spills[0].update_environment_conditions(model.model_time)
    #model.spills[0].tamoc_parameters['depth'] = model.spills[0].start_position[2]
    print("about to start running the model")
    for step in model:
        if step['step_num'] == 0:
            for d in model.spills[0].droplets:
                d.density = 850.
                d.position[2] = 0
#            sp = model.spills[0]
#            print sp.tamoc_parameters
#            sp.update_environment_conditions(model.model_time)
#            print sp.tamoc_parameters
        print(step)
        # model.
