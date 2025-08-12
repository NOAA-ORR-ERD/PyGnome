#!/usr/bin/env python
"""
Script to test GNOME with:

TAMOC - Texas A&M Oilspill Calculator

https://github.com/socolofs/tamoc

This is a very simple environment:

Simple map (no land) and simple current mover (steady uniform current)

Rise velocity and vertical diffusion

But it's enough to see if the coupling with TAMOC works.

"""


import os
import numpy as np
from datetime import datetime, timedelta

from gnome import scripting
from gnome.environment.gridded_objects_base import Variable, Time, Grid_S
from gnome.environment import GridCurrent
from gnome.environment import Wind

from gnome.model import Model
from gnome.maps.map import GnomeMap
from gnome.movers import (RandomMover,
                          TamocRiseVelocityMover,
                          RandomMover3D,
                          SimpleMover,
                          c_GridCurrentMover,
                          CurrentMover,
                          constant_point_wind_mover,
                          PointWindMover)

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput
from gnome.tamoc import tamoc_spill

# define base directory
base_dir = os.path.dirname(__file__)

x, y = np.mgrid[-30:30:61j, -30:30:61j]
y = np.ascontiguousarray(y.T)
x = np.ascontiguousarray(x.T)
# y += np.sin(x) / 1
# x += np.sin(x) / 5
g = Grid_S(node_lon=x,
          node_lat=y)
g.build_celltree()
t = Time.constant_time()
angs = -np.arctan2(y, x)
mag = np.sqrt(x ** 2 + y ** 2)
vx = np.cos(angs) * mag
vy = np.sin(angs) * mag
vx = vx[np.newaxis, :] * 5
vy = vy[np.newaxis, :] * 5

vels_x = Variable(name='v_x', units='m/s', time=t, grid=g, data=vx)
vels_y = Variable(name='v_y', units='m/s', time=t, grid=g, data=vy)
vg = GridCurrent(variables=[vels_y, vels_x], time=t, grid=g, units='m/s')


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    # set up the modeling environment
    start_time = datetime(2016, 9, 18, 1, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=30 * 60,
                  uncertain=False)

    print('adding the map')
    model.map = GnomeMap()  # this is a "water world -- no land anywhere"

    # renderere is only top-down view on 2d -- but it's something
    renderer = Renderer(output_dir=images_dir,
                        image_size=(1024, 768),
                        output_timestep=timedelta(hours=1),
                        )
    renderer.viewport = ((-87.295, 27.795), (-87.705, 28.205))

    print('adding outputters')
    model.outputters += renderer

    # Also going to write the results out to a netcdf file
    netcdf_file = os.path.join(base_dir, 'gulf_tamoc.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file,
                                     which_data='most',
                                     # output most of the data associated with the elements
                                     output_timestep=timedelta(hours=2))

    print("adding Horizontal and Vertical diffusion")

    # Horizontal Diffusion
    #model.movers += RandomMover(diffusion_coef=100000)
    # vertical diffusion (different above and below the mixed layer)
    model.movers += RandomMover3D(vertical_diffusion_coef_above_ml=50,
                                        vertical_diffusion_coef_below_ml=10,
                                        horizontal_diffusion_coef_above_ml=100000,
                                        horizontal_diffusion_coef_below_ml=100,
                                        mixed_layer_depth=10)

    print('adding Rise Velocity')
    # droplets rise as a function of their density and radius
    model.movers += TamocRiseVelocityMover()

    hycom_file = os.path.join(base_dir, 'HYCOM_3d.nc')

    print('adding the 3D current mover')
    #gc = GridCurrent.from_netCDF('HYCOM_3d.nc')
    gc = GridCurrent.from_netCDF(hycom_file)

    #model.movers += GridCurrentMover('HYCOM_3d.nc')
    model.movers += CurrentMover(hycom_file)
#    model.movers += SimpleMover(velocity=(0., 0, 0.))
    model.movers += constant_point_wind_mover(10, 315, units='knots')

    # Wind from a buoy
    #w = Wind(filename='KIKT.osm')
    #model.movers += PointWindMover(w)


    # Now to add in the TAMOC "spill"
    print("Adding TAMOC spill")

    model.spills += tamoc_spill.TamocSpill(release_time=start_time,
                                        start_position=(-87.5, 28.0, 1000),
                                        num_elements=1000,
                                        release_duration=timedelta(days=2),
                                        #end_release_time=start_time + timedelta(days=2),
                                        name='TAMOC plume',
                                        #TAMOC_interval=None,  # how often to re-run TAMOC
                                        )

    #model.spills[0].data_sources['currents'] = gc

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    #model.spills[0].update_environment_conditions(model.model_time)
    #model.spills[0].tamoc_parameters['nbins'] = 20
    print("about to start running the model")
    for step in model:
        #if step['step_num'] == 1:
#             import random
#             for d in model.spills[0].droplets:
#                 d.density= random.randint(800,850)
#                  d.density = 850.
#            print 'running tamoc again'
#            sp = model.spills[0]
#            print sp.tamoc_parameters
#            sp.update_environment_conditions(model.model_time)
#            print sp.tamoc_parameters
#            sp._run_tamoc()
        print(step)
        # model.
