#!/usr/bin/env python

"""
Script to test GNOME with HYCOM data in Mariana Islands region.


"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind

from gnome import scripting

# define base directory
base_dir = os.path.dirname(__file__)

def make_model(images_dir=os.path.join(base_dir,"images")):
    print "initializing the model"

    start_time = datetime(2013, 5, 18, 0)
    model = gnome.model.Model(start_time = start_time,
                              duration = timedelta(days=8),	# 9 day of data in file
                              time_step = 4 * 3600, # 4 hr in seconds
                              uncertain = False,
                              )
    
    mapfile = os.path.join( base_dir, './mariana_island.bna')
    print "adding the map"
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=6, #hours
                                     )
    

    ##
    ## Add teh outputers -- render to images, and save out as netCDF
    ##

    print "adding renderer" 
    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800, 600))
    model.outputters += renderer
    
    # print "adding netcdf output"
    # netcdf_output_file = os.path.join(base_dir,'mariana_output.nc')
    # scripting.remove_netcdf(netcdf_output_file)
    # model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_output_file,
    #                                                         all_data=True)

    ##
    ## Set up the movers:
    ##

    print  "adding a RandomMover:"
    model.movers += gnome.movers.RandomMover(diffusion_coef=10000)
    
    print "adding a simple wind mover:"
    model.movers += gnome.movers.constant_wind_mover(5, 315, units='m/s')
    
    print "adding a current mover:"
    ## this is HYCOM currents
    curr_file=os.path.join( base_dir, r"./HYCOM.nc")
    model.movers += gnome.movers.GridCurrentMover(curr_file)

    ##
    ## Add some spills (sources of elements)
    ##
    print "adding four spill"
    
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=250,
                                            start_position = (145.25, 15.0, 0.0),
                                            release_time = start_time,
                                            )
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=250,
                                            start_position = (146.25, 15.0, 0.0),
                                            release_time = start_time,
                                            )
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=250,
                                            start_position = (145.75, 15.25, 0.0),
                                            release_time = start_time,
                                            )
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=250,
                                            start_position = (145.75, 14.75, 0.0),
                                            release_time = start_time,
                                            )

    return model


if __name__ == "__main__":
    """ if called on its own -- run it """
    from gnome import scripting

    scripting.make_images_dir()
    model = make_model()
    model.full_run(log=True)
    


