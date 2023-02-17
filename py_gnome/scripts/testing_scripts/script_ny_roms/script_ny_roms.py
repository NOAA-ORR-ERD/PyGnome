"""
Script to test how a Property object handles an OpenDAP URL for
curvilinear gridded data

This script uses:
- GridCurrent
- GridCurrentMover
- rendering of GridCurrent using Renderer
"""







import os
from datetime import datetime, timedelta


import gnome.scripting as gs

from gnome import utilities


from gnome.model import Model

# import gnome.utilities.profiledeco as pd

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):

    print('initializing the model')
    start_time = datetime(2012, 10, 25, 0, 1)
    # start_time = datetime(2015, 12, 18, 06, 01)

    # 1 day of data in file
    # 1/2 hr in seconds
    model = Model(start_time=start_time,
                  duration=timedelta(hours=6),
                  time_step=900)

    mapfile = gs.get_datafile(os.path.join(base_dir, 'nyharbor.bna'))

    print('adding the map')
    # TODO: sort out MapFromBna's map_bounds parameter...
    #       it does nothing right now, and the spill is out of bounds'''
    print("loading map:", mapfile)
    model.map = gs.MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

    # draw_ontop can be 'uncertain' or 'forecast'
    # 'forecast' LEs are in black, and 'uncertain' are in red
    # default is 'forecast' LEs draw on top
    renderer = gs.Renderer(mapfile, images_dir, image_size=(1024, 768))
#     renderer.viewport = ((-73.5, 40.5), (-73.1, 40.75))
#     renderer.viewport = ((-122.9, 45.6), (-122.6, 46.0))

    print('adding outputters')
    model.outputters += renderer

    print('adding a spill')
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    spill1 = gs.surface_point_line_spill(num_elements=1000,
                                      start_position=(-74.15,
                                                      40.5,
                                                      0.0),
                                      release_time=start_time)

    model.spills += spill1

    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=50000)

    print('adding a wind mover:')

    model.movers += gs.constant_point_wind_mover(4, 270, units='m/s')

    print('adding a current mover:')

    # url is broken, fix and include the following section
#     url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
# #     cf = roms_field('nos.tbofs.fields.n000.20160406.t00z_sgrid.nc')
#     cf = GridCurrent.from_netCDF(url)
#     renderer.add_grid(cf.grid)
#     renderer.delay = 25
#     u_mover = GridCurrentMover(cf, default_num_method='Euler')
#     model.movers += u_mover
#

    return model


if __name__ == "__main__":
    # NOTE: BNA reading fails with the profiler on!
    #       issue with resizing a numpy array
    #pd.profiler.enable()
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

        print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use()))
    print(datetime.now() - startTime)
    # pd.profiler.disable()
    # pd.print_stats(0.2)
