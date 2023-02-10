#!/usr/bin/env python
"""
Script to test GNOME with subsurface plume spill in a more "real" world
 - Weibull droplet size distribution

Simple map and simple current mover

Rise velocity and vertical diffusion
"""

import os
from datetime import datetime

from gnome import scripting as gs

from gnome.utilities.distributions import WeibullDistribution, UniformDistribution
from gnome.environment import Water
# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = datetime(2012, 10, 25, 0, 1)

    # 1 day of data in file
    model = gs.Model(start_time=start_time,
                  duration=gs.hours(2),
                  time_step=900,
                  )

    mapfile = gs.get_datafile(os.path.join(base_dir, 'nyharbor.bna'))

    print('adding the map')
    '''TODO: sort out MapFromBna's map_bounds parameter...
    it does nothing right now, and the spill is out of bounds'''
    model.map = gs.MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = gs.Renderer(mapfile, images_dir, image_size=(1024, 768))
#     renderer.viewport = ((-73.5, 40.5), (-73.1, 40.75))
#     renderer.viewport = ((-122.9, 45.6), (-122.6, 46.0))

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_ny_plume.nc')
    gs.remove_netcdf(netcdf_file)

    model.outputters += gs.NetCDFOutput(netcdf_file, which_data='all')

    print('adding the spills')

    end_time = start_time + model.duration

    oil_name = 'oil_ans_mp'
    wd = WeibullDistribution(alpha=1.8,
                             lambda_=.00456,
                             min_=.0002)  # 200 micron min

    spill = gs.subsurface_spill(num_elements=1000,
                                   start_position=(-74.15,
                                                   40.5,
                                                   7.2),
                                   release_time=start_time,
                                   distribution=wd,
                                   amount=90,  # default volume_units=m^3
                                   units='m^3',
                                   end_release_time=end_time,
                                   substance = gs.GnomeOil(oil_name),
                                   )

    model.spills += spill

    print('adding environment object:')
    model.environment += [Water(temperature=25.0 + 273.15),] 
    
    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=50000)

    print('adding a RiseVelocityMover:')
    model.movers += gs.RiseVelocityMover()

    print('adding a RandomMover3D:')
    model.movers += gs.RandomMover3D(vertical_diffusion_coef_above_ml=5,
                                  vertical_diffusion_coef_below_ml=.11,
                                  mixed_layer_depth=10)

    # the url is broken, update and include the following four lines
#     url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
#     gc = GridCurrent.from_netCDF(url)
#     u_mover = gs.GridCurrentMover(gc, default_num_method='RK2')
#     model.movers += u_mover

    print('adding a wind mover:')

    w_mover = gs.constant_point_wind_mover(15, 270, units='knot')
    model.movers += w_mover

    return model


if __name__ == "__main__":
    startTime = datetime.now()
    gs.make_images_dir()
    model = make_model()
    print("doing full run")
    rend = model.outputters[0]
#     field = rend.grids[0]
#     rend.graticule.set_DMS(True)
    for step in model:
        if step['step_num'] == 0:
            rend.set_viewport(((-74.25, 40.4), (-73.9, 40.6)))
        print(f"step: {step['step_num']}")
    print(f"run took: {datetime.now() - startTime}")
