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
from gnome.spills import point_line_release_spill
from gnome.movers import RandomMover, constant_wind_mover, GridCurrentMover

from gnome.outputters import Renderer
from gnome.environment import GridCurrent
from gnome.movers.py_current_movers import GridCurrentMover
import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print 'initializing the model'

    start_time = datetime(2016, 4, 5, 18, 0)


    model = Model(start_time=start_time,
                  duration=timedelta(hours=12),
                  time_step=.25 * 3600)

    mapfile = (os.path.join(base_dir, 'coast.bna'))

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
    spill1 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73888,
                                                      27.5475,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))
    spill2 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73888,
                                                      27.545,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))
    spill3 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73888,
                                                      27.5425,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))
    spill4 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73988,
                                                      27.5475,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))

    spill5 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73988,
                                                      27.5450,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))

    spill6 = point_line_release_spill(num_elements=500,
                                      start_position=(-82.73988,
                                                      27.5425,
                                                      0.0),
                                      release_time=start_time,
                                      end_release_time=start_time + timedelta(hours=24))


    model.spills += spill1
    model.spills += spill2
    model.spills += spill3
    model.spills += spill4
    model.spills += spill5
    model.spills += spill6
    model.spills._spill_container.spills.remove(0)


    print 'adding a current mover:'

    fn = 'nos.tbofs.fields.n000.20160406.t00z_sgrid.nc'
    # fn = 'dbofs_newFormat.nc'

    cf = GridCurrent.from_netCDF(filename=fn)
    u_mover = GridCurrentMover(cf, extrapolate=True)
    # u_mover = GridCurrentMover(fn)
    renderer.add_grid(cf.grid)
#     renderer.add_vec_prop(cf)
    model.movers += u_mover

    # curr_file = get_datafile(os.path.join(base_dir, 'COOPSu_CREOFS24.nc'))
    # c_mover = GridCurrentMover(curr_file)
    # model.movers += c_mover

    return model


if __name__ == "__main__":
    pd.profiler.enable()
    startTime = datetime.now()
    scripting.make_images_dir()
    model = make_model()
    print "doing full run"
    rend = model.outputters[0]
    for step in model:
        if step['step_num'] == 0:
            rend.set_viewport(((-82.8, 27.475), (-82.7, 27.575)))
#         if step['step_num'] == 0:
#             rend.set_viewport(((-83.5, 27.0), (-82.1, 28.0)))

        print "step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use())
    print datetime.now() - startTime
    pd.profiler.disable()
    pd.print_stats(0.2)
