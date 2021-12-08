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

from gnome import scripting
from gnome import utilities
from gnome.basic_types import datetime_value_2d, numerical_methods

from gnome.utilities.remote_data import get_datafile

#from gnome.spills.elements import plume
from gnome.spills.substance import GnomeOil
from gnome.spills.initializers import plume_initializers
from gnome.utilities.distributions import WeibullDistribution, UniformDistribution
from gnome.maps import MapFromBNA
from gnome.model import Model
from gnome.environment import GridCurrent
from gnome.spills import surface_point_line_spill
from gnome.scripting import subsurface_plume_spill
from gnome.movers import (RandomMover,
                          RiseVelocityMover,
                          RandomMover3D,
                          SimpleMover)

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput
from gnome.movers.py_current_movers import PyCurrentMover
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = datetime(2012, 10, 25, 0, 1)
    # start_time = datetime(2015, 12, 18, 06, 01)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(hours=2),
                  time_step=900)

    mapfile = get_datafile(os.path.join(base_dir, 'nyharbor.bna'))

    print('adding the map')
    '''TODO: sort out MapFromBna's map_bounds parameter...
    it does nothing right now, and the spill is out of bounds'''
    model.map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
#     renderer.viewport = ((-73.5, 40.5), (-73.1, 40.75))
#     renderer.viewport = ((-122.9, 45.6), (-122.6, 46.0))

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_ny_plume.nc')
    scripting.remove_netcdf(netcdf_file)

    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print('adding two spills')
    # Break the spill into two spills, first with the larger droplets
    # and second with the smaller droplets.
    # Split the total spill volume (100 m^3) to have most
    # in the larger droplet spill.
    # Smaller droplets start at a lower depth than larger

    end_time = start_time + model.duration
#     wd = WeibullDistribution(alpha=1.8,
#                              lambda_=.00456,
#                              min_=.0002)  # 200 micron min
# 
#     spill = subsurface_plume_spill(num_elements=10,
#                                    start_position=(-74.15,
#                                                    40.5,
#                                                    7.2),
#                                    release_time=start_time,
#                                    distribution=wd,
#                                    amount=90,  # default volume_units=m^3
#                                    units='m^3',
#                                    end_release_time=end_time,
#                                    density=600)
# 
#     model.spills += spill

#     wd = WeibullDistribution(alpha=1.8,
#                              lambda_=.00456,
#                              max_=.0002)  # 200 micron max

   # oil_name = 'ALASKA NORTH SLOPE (MIDDLE PIPELINE, 1997)'
   # use sample oil or download file from adios database
    oil_name = 'oil_ans_mp'
    
    wd = UniformDistribution(low=.0002,
                             high=.0002)

    spill = surface_point_line_spill(num_elements=10, amount=90,
                                     units='m^3',
                                     start_position=(-74.15,
                                                     40.5,
                                                     7.2),
                                     release_time=start_time,
                                     substance = GnomeOil(oil_name,initializers=plume_initializers(distribution=wd))
                                     #element_type=plume(distribution=wd,
                                                        #substance_name='ALASKA NORTH SLOPE (MIDDLE PIPELINE, 1997)')
                                     )
    model.spills += spill

    print('adding a RandomMover:')
    model.movers += RandomMover(diffusion_coef=50000)

    print('adding a RiseVelocityMover:')
    model.movers += RiseVelocityMover()

    print('adding a RandomMover3D:')
#     model.movers += RandomMover3D(vertical_diffusion_coef_above_ml=5,
#                                         vertical_diffusion_coef_below_ml=.11,
#                                         mixed_layer_depth=10)

    # the url is broken, update and include the following four lines
#     url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
#     gc = GridCurrent.from_netCDF(url)
#     u_mover = PyCurrentMover(gc, default_num_method='RK2')
#     model.movers += u_mover

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

    print('adding a simple mover:')
#     s_mover = SimpleMover(velocity=(0.0, -.3, 0.0))
#     model.movers += s_mover

    return model


if __name__ == "__main__":
#     pd.profiler.enable()
    startTime = datetime.now()
    scripting.make_images_dir()
    model = make_model()
    print("doing full run")
    rend = model.outputters[0]
#     field = rend.grids[0]
#     rend.graticule.set_DMS(True)
    for step in model:
        if step['step_num'] == 0:
            rend.set_viewport(((-74.25, 40.4), (-73.9, 40.6)))

        print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use()))
    print(datetime.now() - startTime)
#     pd.profiler.disable()
#     pd.print_stats(0.2)
