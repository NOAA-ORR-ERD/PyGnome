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

from gnome.environment.property_classes import WindTS

from gnome.outputters import Renderer
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2012, 10, 25, 0, 1)
    # start_time = datetime(2015, 12, 18, 06, 01)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(hours=24),
                  time_step=900)

    mapfile = get_datafile(os.path.join(base_dir, 'nyharbor.bna'))

    print 'adding the map'
    '''TODO: sort out MapFromBna's map_bounds parameter...
    it does nothing right now, and the spill is out of bounds'''
    model.map = MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = Renderer(mapfile, images_dir, image_size=(1024, 768))
#     renderer.viewport = ((-73.5, 40.5), (-73.1, 40.75))
#     renderer.viewport = ((-122.9, 45.6), (-122.6, 46.0))

    print 'adding outputters'
    model.outputters += renderer

    print 'adding a spill'
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill1 = point_line_release_spill(num_elements=1000,
                                      start_position=(-74.15,
                                                      40.5,
                                                      0.0),
                                      release_time=start_time)

    model.spills += spill1

    print 'adding a RandomMover:'
#     model.movers += RandomMover(diffusion_coef=50000)

    print 'adding a wind mover:'

    series = []
    for i in [(1, (5, 90)), (7, (5, 180)), (13, (5, 270)), (19, (5, 0)), (25, (5, 90))]:
        series.append((start_time + timedelta(hours=i[0]), i[1]))

    wind1 = WindTS.constant_wind('wind1', 5, 270, 'knots')
    wind2 = WindTS(timeseries = series, units='knots', extrapolate=True)

    wind = Wind(timeseries=series, units='knots')

    model.movers += WindMover(wind=wind2)
#     model.movers += PointWindMover(wind)

#     print 'adding a current mover:'
#
#     url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
# #     cf = roms_field('nos.tbofs.fields.n000.20160406.t00z_sgrid.nc')
#     cf = roms_field(url)
#     cf.set_appearance(on=True)
#     renderer.grids += [cf]
#     renderer.delay = 25
#     u_mover = UGridCurrentMover(cf)
#     model.movers += u_mover

    # curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))
    # c_mover = CurrentMover(curr_file)
    # model.movers += c_mover

    return model


if __name__ == "__main__":
    pd.profiler.enable()
    startTime = datetime.now()
    scripting.make_images_dir()
    model = make_model()
    print "doing full run"
    rend = model.outputters[0]
#     field = rend.grids[0]
#     rend.graticule.set_DMS(True)
    for step in model:
        if step['step_num'] == 0:
            rend.set_viewport(((-74.25, 40.4), (-73.9, 40.6)))

        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    print datetime.now() - startTime
    pd.profiler.disable()
    pd.print_stats(0.2)
