"""
Script to test GNOME with chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA
"""

import os
from datetime import datetime, timedelta


from gnome import scripting
from gnome import utilities

from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.maps import MapFromBNA
from gnome.environment import Environment
from gnome.spills import surface_point_line_spill
from gnome.movers import RandomMover, constant_point_wind_mover, c_GridCurrentMover, IceAwareRandomMover

from gnome.environment import IceAwareCurrent, IceAwareWind, GridCurrent
from gnome.movers.py_wind_movers import WindMover
from gnome.movers.py_current_movers import CurrentMover

from gnome.outputters import Renderer, NetCDFOutput
import gnome.utilities.profiledeco as pd
from gnome.environment.environment_objects import IceVelocity

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = datetime(1985, 1, 1, 13, 31)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(days=4),
                  time_step=7200)

#     mapfile = get_datafile(os.path.join(base_dir, 'ak_arctic.bna'))
    mapfile = get_datafile('arctic_coast3.bna')

    print('adding the map')
    model.map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    print('adding a spill')
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
#     spill1 = surface_point_line_spill(num_elements=10000,
#                                       start_position=(-163.75,
#                                                       69.75,
#                                                       0.0),
#                                       release_time=start_time)
#
    spill1 = surface_point_line_spill(num_elements=50000,
                                      start_position=(196.25,
                                                      69.75,
                                                      0.0),
                                      release_time=start_time)

    model.spills += spill1
#     model.spills += spill2

    print('adding a wind mover:')

#     model.movers += constant_point_wind_mover(0.5, 0, units='m/s')

    print('adding a current mover:')

#     fn = ['arctic_avg2_0001_gnome.nc',
#           'arctic_avg2_0002_gnome.nc']

    fn = [get_datafile(os.path.join(base_dir, 'arctic_avg2_0001_gnome.nc')),
          get_datafile(os.path.join(base_dir, 'arctic_avg2_0002_gnome.nc')),
          ]

    # filelist is not working
    fn = get_datafile(os.path.join(base_dir, 'arctic_avg2_0001_gnome.nc'))
#     fn = ['C:\\Users\\jay.hennen\\Documents\\Code\\pygnome\\py_gnome\\scripts\\script_TAP\\arctic_avg2_0001_gnome.nc',
#           'C:\\Users\\jay.hennen\\Documents\\Code\\pygnome\\py_gnome\\scripts\\script_TAP\\arctic_avg2_0002_gnome.nc']

    gt = {'node_lon': 'lon',
          'node_lat': 'lat'}
#     fn='arctic_avg2_0001_gnome.nc'

    wind_method = 'Euler'
    method = 'RK2'
    print('adding outputters')

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
    model.outputters += renderer
    netcdf_file = os.path.join(base_dir, str(model.time_step / 60) + method + '.nc')
    scripting.remove_netcdf(netcdf_file)

    print('adding movers')
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')


    print('loading entire current data')
    ice_aware_curr = IceAwareCurrent.from_netCDF(filename=fn,
                                                 grid_topology=gt)

#     env1 = get_env_from_netCDF(filename)
#     mov = GridCurrentMover.from_netCDF(filename)

    ice_aware_curr.ice_velocity.variables[0].dimension_ordering = ['time', 'x', 'y']
    ice_aware_wind = IceAwareWind.from_netCDF(filename=fn,
                                              ice_velocity=ice_aware_curr.ice_velocity,
                                              ice_concentration=ice_aware_curr.ice_concentration,
                                              grid=ice_aware_curr.grid)

    curr = GridCurrent.from_netCDF(filename=fn)
#     GridCurrent.is_gridded()

#     import pprint as pp
#     from gnome.utilities.orderedcollection import OrderedCollection
#     model.environment = OrderedCollection(dtype=Environment)
#     model.environment.add(ice_aware_curr)
#     from gnome.environment import WindTS

    print('loading entire wind data')

#     i_c_mover = GridCurrentMover(current=ice_aware_curr)
#     i_c_mover = GridCurrentMover(current=ice_aware_curr, default_num_method='Euler')
   # i_c_mover = GridCurrentMover(current=ice_aware_curr, default_num_method=method, extrapolate=True)
    i_c_mover = CurrentMover(current=ice_aware_curr, default_num_method=method)
    i_w_mover = WindMover(wind=ice_aware_wind, default_num_method=wind_method)

    i_c_mover.current.grid.extrapolation_is_allowed = True

#     ice_aware_curr.grid.node_lon = ice_aware_curr.grid.node_lon[:]-360
#     ice_aware_curr.grid.build_celltree()
    model.movers += i_c_mover
    model.movers += i_w_mover

    print('adding an IceAwareRandomMover:')
    model.movers += IceAwareRandomMover(ice_concentration=ice_aware_curr.ice_concentration,
                                        diffusion_coef=1000)
#     renderer.add_grid(ice_aware_curr.grid)
#     renderer.add_vec_prop(ice_aware_curr)


    # curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))
    # c_mover = c_GridCurrentMover(curr_file)
    # model.movers += c_mover
#     model.environment.add(WindTS.constant(10, 300))
#     print('Saving')
#     model.environment[0].ice_velocity.variables[0].serialize()
#     IceVelocity.deserialize(model.environment[0].ice_velocity.serialize())
#     model.save('.')
#     from gnome.persist.save_load import load
#     print('Loading')
#     model2 = load('./Model.zip')

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    print("doing full run")
#     rend = model.outputters[0]
#     rend.graticule.set_DMS(True)
    startTime = datetime.now()
#     pd.profiler.enable()
    for step in model:
#         if step['step_num'] == 0:
#             rend.set_viewport(((-165, 69.25), (-162.5, 70)))
#         if step['step_num'] == 0:
#             rend.set_viewport(((-175, 65), (-160, 70)))
        print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use()))
    print(datetime.now() - startTime)
#     pd.profiler.disable()
#     pd.print_stats(0.1)
