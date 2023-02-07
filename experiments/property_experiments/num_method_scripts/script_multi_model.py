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

from gnome.maps import MapFromBNA
from gnome.environment import Wind
from gnome.spills import point_line_release_spill
from gnome.movers import RandomMover, constant_wind_mover, GridCurrentMover

from gnome.movers.py_wind_movers import WindMover
from gnome.environment.property_classes import GridCurrent
from gnome.movers.py_current_movers import CurrentMover

from gnome.outputters import Renderer, NetCDFOutput
from gnome.environment.vector_field import ice_field
from gnome.movers import PyIceMover
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_models():
    print('initializing the model')

    # start_time = datetime(2015, 12, 18, 06, 01)

    # 1 day of data in file
    # 1/2 hr in seconds
    models = []
    start_time = datetime(2012, 10, 27, 0, 30)
    duration_hrs=23
    time_step=450
    num_steps = duration_hrs * 3600 / time_step
    names = [
             'Euler',
             'Trapezoid',
             'RK4',
             ]

    mapfile = get_datafile(os.path.join(base_dir, 'long_beach.bna'))
    print('gen map')
    map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds
    fn = ('00_dir_roms_display.ncml.nc4')
    curr = GridCurrent.from_netCDF(filename=fn)
    models = []
    for method in names:

        mod = Model(start_time=start_time,
                    duration=timedelta(hours=duration_hrs),
                    time_step=time_step)

        mod.map = map
        spill = point_line_release_spill(num_elements=1000,
                                         start_position=(-74.1,
                                                      39.7525,
                                                      0.0),
                                         release_time=start_time)
        mod.spills += spill
        mod.movers += RandomMover(diffusion_coef=100)
        mod.movers += CurrentMover(current=curr, default_num_method=method)

        images_dir = method + '-' + str(time_step / 60) + 'min-' + str(num_steps) + 'steps'
        renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
        renderer.delay = 25
#         renderer.add_grid(curr.grid)
        mod.outputters += renderer


        netCDF_fn = os.path.join(base_dir, images_dir + '.nc')
        mod.outputters += NetCDFOutput(netCDF_fn, which_data='all')
        models.append(mod)

    print('returning models')
    return models


if __name__ == "__main__":
    models = make_models()
#     for m in models:
#         scripting.make_images_dir(m.outputters)
    print("doing full run")
#     field = rend.grids[0]
#     rend.graticule.set_DMS(True)
    for model in models:
        rend = model.outputters[0]
        startTime = datetime.now()
        pd.profiler.enable()
        for step in model:
            if step['step_num'] == 0:
                rend.set_viewport(((-74.2, 39.75), (-74.05, 39.85)))

            print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                                  utilities.get_mem_use()))
        print(datetime.now() - startTime)
        pd.profiler.disable()
        pd.print_stats(0.1)
        pd.clear_stats()
        print('\n-----------------------------------------------\n')
