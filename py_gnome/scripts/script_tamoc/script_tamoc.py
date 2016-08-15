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
from datetime import datetime, timedelta

from gnome import scripting
from gnome.spill.elements import plume
from gnome.utilities.distributions import WeibullDistribution

from gnome.model import Model
from gnome.map import GnomeMap
from gnome.spill import point_line_release_spill
from gnome.scripting import subsurface_plume_spill
from gnome.movers import (RandomMover,
                          RiseVelocityMover,
                          RandomVerticalMover,
                          SimpleMover)

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput
from gnome.tamoc import tamoc

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    # set up the modeling environment
    start_time = datetime(2004, 12, 31, 13, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=30 * 60,
                  uncertain=False)

    print 'adding the map'
    model.map = GnomeMap()  # this is a "water world -- no land anywhere"

    # renderere is only top-down view on 2d -- but it's something
    renderer = Renderer(output_dir=images_dir,
                        size=(1024, 768),
                        output_timestep=timedelta(hours=1),
                        )
    renderer.viewport = ((-76.5, 37.), (-75.5, 38.))

    print 'adding outputters'
    model.outputters += renderer

    # Also going to write the results out to a netcdf file
    netcdf_file = os.path.join(base_dir, 'script_plume.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file,
                                     which_data='most',
                                     # output most of the data associated with the elements
                                     output_timestep=timedelta(hours=2))

    print "adding Horizontal and Vertical diffusion"

    # Horizontal Diffusion
    model.movers += RandomMover(diffusion_coef=50000)
    # vertical diffusion (different above and below the mixed layer)
    model.movers += RandomVerticalMover(vertical_diffusion_coef_above_ml=5,
                                        vertical_diffusion_coef_below_ml=.11,
                                        mixed_layer_depth=10)

    print 'adding Rise Velocity'
    # droplets rise as a function of their density and radius
    model.movers += RiseVelocityMover()

    print 'adding a stady uniform current'
    # This is .3 m/s south
    model.movers += SimpleMover(velocity=(0.0, -.3, 0.0))

    # Now to add in the TAMOC "spill"
    print "Adding TAMOC spill"

    end_time = start_time + timedelta(hours=24)

    tamoc.TamocSpill(release_time=start_time,
                     start_position=(-76, 37.5, 1000),
                     num_elements=10000,
                     end_release_time=start_time + timedelta(days=1),
                     name='TAMOC plume',
                     TAMOC_interval=None,  # how often to re-run TAMOC
                     )

    model.spills += spill

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    print "about to start running the model"
    for step in model:
        print step
        #model.
