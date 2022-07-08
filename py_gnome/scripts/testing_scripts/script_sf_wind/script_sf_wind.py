#!/usr/bin/env python
"""
Script to test GNOME with san francisco bay data (NWS wind data)
Uses NonweatheringSubstance. Sets windage range and uncertainty values
"""

import os
from datetime import datetime, timedelta

from gnome import scripting
from gnome.spills.substance import NonWeatheringSubstance
#from gnome.spills.elements import floating

from gnome.utilities.remote_data import get_datafile

from gnome.model import Model
from gnome.maps import MapFromBNA
from gnome.spills import surface_point_line_spill
from gnome.movers import c_GridWindMover

from gnome.outputters import Renderer
from gnome.outputters import NetCDFOutput

# define base directory
base_dir = os.path.dirname(__file__)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')
    start_time = datetime(2006, 3, 31, 20, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=3), time_step=30 * 60,
                  uncertain=True)

    print('adding the map')
    mapfile = get_datafile(os.path.join(base_dir, 'coastSF.bna'))
    model.map = MapFromBNA(mapfile, refloat_halflife=1)  # seconds

    renderer = Renderer(mapfile, images_dir, image_size=(800, 600),
                        draw_ontop='forecast')
    renderer.viewport = ((-124.5, 37.), (-120.5, 39))

    print('adding outputters')
    model.outputters += renderer

    netcdf_file = os.path.join(base_dir, 'script_sf_bay.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += NetCDFOutput(netcdf_file, which_data='all')

    print('adding a spill')
    spill = surface_point_line_spill(num_elements=1000,
                                     start_position=(-123.57152, 37.369436,
                                                     0.0),
                                     release_time=start_time,
                                     substance=NonWeatheringSubstance(windage_range=(0.01,.04))
                                     #element_type=floating(windage_range=(0.01,
                                     #                                     0.04)
                                     #                      )
                                     )
    model.spills += spill

    # print 'adding a RandomMover:'
    # r_mover = gnome.movers.RandomMover(diffusion_coef=50000)
    # model.movers += r_mover

    print('adding a grid wind mover:')
    wind_file = get_datafile(os.path.join(base_dir, 'WindSpeedDirSubset.nc'))
    topology_file = get_datafile(os.path.join(base_dir,
                                              'WindSpeedDirSubsetTop.dat'))
    w_mover = c_GridWindMover(wind_file, topology_file)

    # w_mover.uncertain_time_delay = 6
    # w_mover.uncertain_duration = 6
    w_mover.uncertain_speed_scale = 1
    w_mover.uncertain_angle_scale = 0.2  # default is .4
    w_mover.wind_scale = 2

    model.movers += w_mover

    return model

if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    model.full_run()
