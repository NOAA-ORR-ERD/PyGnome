#!/usr/bin/env python
"""
Script to test GNOME with HYCOM data in Mariana Islands region.
"""






import os

import gnome.scripting as gs

from gnome import utilities

from gnome.maps.map import ParamMap


NUM_ELEMENTS = 10000

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(img_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    start_time = "2013-05-18T00:00:00"

    model = gs.Model(start_time=start_time, duration=gs.days(3),
                     time_step=3600, uncertain=False)

    print('adding the map')
    p_map = model.map = ParamMap(center=(0, 0),
                                 distance=20,
                                 bearing=20,
                                 units='km',
                                 )  # hours

    #
    # Add the outputters -- render to images, and save out as netCDF
    #

    print('adding renderer')
    rend = gs.Renderer(output_dir=img_dir,
                       image_size=(800, 600),
                       map_BB=p_map.get_map_bounds(),
                       land_polygons=p_map.get_land_polygon(),
                       )

    rend.graticule.set_DMS(True)
    model.outputters += rend

#                                 draw_back_to_fore=True)

    # print "adding netcdf output"
    # netcdf_output_file = os.path.join(base_dir,'mariana_output.nc')
    # scripting.remove_netcdf(netcdf_output_file)
    # model.outputters += NetCDFOutput(netcdf_output_file, which_data='all')

    #
    # Set up the movers:
    #

    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=100000)

    print('adding a simple wind mover:')
    model.movers += gs.constant_point_wind_mover(10, 225, units='m/s')

    print('adding a current mover:')

#     # # this is HYCOM currents
#     curr_file = get_datafile(os.path.join(base_dir, 'HYCOM.nc'))
#     model.movers += c_GridCurrentMover(curr_file,
#                                      num_method=numerical_methods.euler);

    # #
    # # Add some spills (sources of elements)
    # #

    print('adding four spill')
    model.spills += gs.surface_point_line_spill(num_elements=NUM_ELEMENTS // 4,
                                                start_position=(0.0,
                                                                0.0,
                                                                0.0),
                                                release_time=start_time)

    return model


if __name__ == '__main__':
    gs.make_images_dir()
    model = make_model()
    rend = model.outputters[0]
    for step in model:
#         rend.zoom(0.9)
        if (step['step_num'] == 33):
            pass

        print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use()))
    # model.full_run(log=True)