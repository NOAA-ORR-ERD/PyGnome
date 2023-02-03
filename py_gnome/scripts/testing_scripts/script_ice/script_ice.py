#!/usr/bin/env python

"""
Example of using gnome in "ice infested waters"

With the ice and current data coming from a ROMS coupled
ocean-ice model.
"""


# The gnome.scripting module provides most of what you need for basic scripts
import gnome.scripting as gs

import os
# define base directory
base_dir = os.path.dirname(__file__)


# gs.PrintFinder()


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = "1985-01-01T13:31"

    model = gs.Model(start_time=start_time,
                     duration=gs.days(2),
                     time_step=gs.hours(1))

    mapfile = gs.get_datafile(os.path.join(base_dir, 'ak_arctic.bna'))

    print('adding the map')
    model.map = gs.MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    print('adding outputters')
    renderer = gs.Renderer(mapfile, images_dir, image_size=(1024, 768))
    renderer.set_viewport(((-165, 69), (-161.5, 70)))

    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_ice.nc')
    gs.remove_netcdf(netcdf_file)
    model.outputters += gs.NetCDFOutput(netcdf_file,
                                        which_data='all')

    print('adding a spill')
    # For a subsurfce spill, you would need to add vertical movers:
    # - gs.RiseVelocityMover
    # - gs.RandomMover3D
    spill1 = gs.surface_point_line_spill(num_elements=1000,
                                         start_position=(-163.75,
                                                         69.75,
                                                         0.0),
                                         release_time=start_time)

    model.spills += spill1

    print('adding the ice movers')
    print('getting the datafiles')
    fn = [gs.get_datafile((os.path.join(base_dir,'arctic_avg2_0001_gnome.nc'))),
          gs.get_datafile((os.path.join(base_dir,'arctic_avg2_0002_gnome.nc'))),
          ]

    gt = {'node_lon': 'lon',
          'node_lat': 'lat'}

    ice_aware_curr = gs.IceAwareCurrent.from_netCDF(filename=fn,
                                                    grid_topology=gt)
    ice_aware_wind = gs.IceAwareWind.from_netCDF(filename=fn,
                                                 grid=ice_aware_curr.grid,)
    i_c_mover = gs.CurrentMover(current=ice_aware_curr)
    i_w_mover = gs.WindMover(wind=ice_aware_wind)

    # shifting to -360 to 0 longitude
    ice_aware_curr.grid.node_lon = ice_aware_curr.grid.node_lon[:] - 360
    model.movers += i_c_mover
    model.movers += i_w_mover

    print('adding an Ice RandomMover:')
    model.movers += gs.IceAwareRandomMover(ice_concentration=ice_aware_curr.ice_concentration,
                                           diffusion_coef=50000)


    # to visualize the grid and currents
#     renderer.add_grid(ice_aware_curr.grid)
#     renderer.add_vec_prop(ice_aware_curr)

    return model


if __name__ == "__main__":
    # gs.set_verbose()
    gs.make_images_dir()
    model = make_model()
    print("doing full run")
    startTime = gs.now()
    for step in model:
        print("step: %.4i" % (step['step_num']))
    print("it took %s to run" % (gs.now() - startTime))
