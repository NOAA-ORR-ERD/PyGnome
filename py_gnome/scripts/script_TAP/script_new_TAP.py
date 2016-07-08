"""
Script to test GNOME with chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA
"""

import os
from datetime import datetime, timedelta

import numpy as np

from gnome import scripting
from gnome import utilities
from gnome.basic_types import datetime_value_2d, numerical_methods

from gnome.utilities.remote_data import get_datafile

from gnome.model import Model

from gnome.map import MapFromBNA
from gnome.environment import Wind
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, constant_wind_mover, GridCurrentMover, IceAwareRandomMover

from gnome.movers.py_wind_movers import PyWindMover
from gnome.environment.property_classes import WindTS, IceAwareCurrent, IceAwareWind, GridWind
from gnome.movers.py_current_movers import PyGridCurrentMover

from gnome.outputters import Renderer, NetCDFOutput
from gnome.environment.vector_field import ice_field
from gnome.movers import PyIceMover
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(1985, 1, 1, 13, 31)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(days=4),
                  time_step=3600*2)

#     mapfile = get_datafile(os.path.join(base_dir, 'ak_arctic.bna'))
    mapfile = get_datafile(os.path.join(base_dir, 'arctic_coast3.bna'))

    print 'adding the map'
    model.map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
#     spill1 = point_line_release_spill(num_elements=10000,
#                                       start_position=(-163.75,
#                                                       69.75,
#                                                       0.0),
#                                       release_time=start_time)
#
    spill1 = point_line_release_spill(num_elements=100,
                                      start_position=(196.25,
                                                      69.75,
                                                      0.0),
                                      release_time=start_time)

    model.spills += spill1
#     model.spills += spill2

    print 'adding a wind mover:'

#     model.movers += constant_wind_mover(0.5, 0, units='m/s')

    print 'adding a current mover:'

    fn = ['arctic_avg2_0001_gnome.nc',
          'arctic_avg2_0002_gnome.nc']

    gt = {'node_lon':'lon',
          'node_lat':'lat'}
#     fn='arctic_avg2_0001_gnome.nc'

    wind_method = 'Euler'
    method = 'Trapezoid'
    print 'adding outputters'

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
#     renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
#     model.outputters += renderer
    netcdf_file = os.path.join(base_dir, str(model.time_step/60) + method + '.nc')
    scripting.remove_netcdf(netcdf_file)

    print 'adding movers'
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    ice_aware_curr = IceAwareCurrent.from_netCDF(filename=fn,
                                                 grid_topology=gt)
    ice_aware_wind = IceAwareWind.from_netCDF(filename=fn,
                                              ice_var = ice_aware_curr.ice_var,
                                              ice_conc_var=ice_aware_curr.ice_conc_var,
                                              grid = ice_aware_curr.grid,)

#     i_c_mover = PyGridCurrentMover(current=ice_aware_curr)
#     i_c_mover = PyGridCurrentMover(current=ice_aware_curr, default_num_method='Euler')
    i_c_mover = PyGridCurrentMover(current=ice_aware_curr, default_num_method=method)
    i_w_mover = PyWindMover(wind = ice_aware_wind, default_num_method=wind_method)

#     ice_aware_curr.grid.node_lon = ice_aware_curr.grid.node_lon[:]-360
#     ice_aware_curr.grid.build_celltree()
    model.movers += i_c_mover
    model.movers += i_w_mover

    print 'adding an IceAwareRandomMover:'
    model.movers += IceAwareRandomMover(ice_conc_var = ice_aware_curr.ice_conc_var,
                                        diffusion_coef=1000)
#     renderer.add_grid(ice_aware_curr.grid)
#     renderer.add_vec_prop(ice_aware_curr)


#     renderer.set_viewport(((-190.9, 60), (-72, 89)))
    # curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))
    # c_mover = GridCurrentMover(curr_file)
    # model.movers += c_mover

    return model


if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    print "doing full run"
#     rend = model.outputters[0]
#     rend.graticule.set_DMS(True)
    startTime = datetime.now()
    pd.profiler.enable()
    for step in model:
#         if step['step_num'] == 0:
#             rend.set_viewport(((-165, 69.25), (-162.5, 70)))
#         if step['step_num'] == 0:
#             rend.set_viewport(((-175, 65), (-160, 70)))
        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    print datetime.now() - startTime
    pd.profiler.disable()
    pd.print_stats(0.1)
