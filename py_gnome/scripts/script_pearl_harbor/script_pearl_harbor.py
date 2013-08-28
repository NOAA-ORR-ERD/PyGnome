#!/usr/bin/env python

"""
Script to test GNOME with CH3D data for Pearl Harbor.

Returns an empty model if it cannot download the Pearl Harbor maps or datafiles
"""

import os
import shutil
from datetime import datetime, timedelta
from urllib2 import HTTPError

import numpy as np

import gnome
from gnome.environment import Wind
from gnome.utilities.remote_data import get_datafile

from gnome import scripting

# define base directory
base_dir = os.path.dirname(__file__)

def make_model(images_dir=os.path.join(base_dir,"images")):
    print "initializing the model"

    start_time = datetime(2013, 1, 1, 1)	# data starts at 1:00 rather than 0:00
    model = gnome.model.Model(start_time = start_time,
                              duration = timedelta(days=1),	# years worth of data in file
                              time_step = 900, # 15 minutes in seconds
                              uncertain = False,
                              )

    try:    
        mapfile = get_datafile( os.path.join( base_dir, './pearl_harbor.bna'))
    except HTTPError:
        print("Could not download Pearl Harbour data from server - returning empty model")
        return model
            
    print "adding the map"
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=1, #hours
                                     )
    

    ##
    ## Add the outputters -- render to images, and save out as netCDF
    ##

    print "adding renderer and netcdf output"
    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800, 600))
    model.outputters += renderer
    
    netcdf_output_file = os.path.join(base_dir,'pearl_harbor_output.nc')
    scripting.remove_netcdf(netcdf_output_file)
    model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_output_file,
                                                            all_data=True)

    ##
    ## Set up the movers:
    ##

    print  "adding a random mover:"
    model.movers += gnome.movers.RandomMover(diffusion_coef=10000)
    
    print "adding a wind mover:"
    #model.movers += gnome.movers.constant_wind_mover(7, 315, units='m/s')
    
    series = np.zeros((3,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time,                      ( 4,   180) )
    series[1] = (start_time+timedelta(hours=12),  ( 2,   270) )
    series[2] = (start_time+timedelta(hours=24),  ( 4,   180) )
    
    
    w_mover = gnome.movers.WindMover( Wind(timeseries=series,units='knots') )
    model.movers += w_mover
    model.environment += w_mover.wind
    
    print "adding a current mover:"
    ## this is CH3D currents
    curr_file=os.path.join( base_dir, r"./ch3d2013.nc")
    topology_file=os.path.join( base_dir, r"./PearlHarborTop.dat")
    model.movers += gnome.movers.GridCurrentMover(curr_file,topology_file)

    ##
    ## Add a spill (sources of elements)
    ##
    print "adding spill"
    
    model.spills += gnome.spill.PointSourceSurfaceRelease(num_elements=1000,
                                            start_position = (-157.97064, 21.331524, 0.0),
                                            release_time = start_time,
                                            )

    return model


if __name__ == "__main__":
    """ if called on its own -- run it """
    from gnome import scripting

    scripting.make_images_dir()
    model = make_model()
    model.full_run(log=True)
    


