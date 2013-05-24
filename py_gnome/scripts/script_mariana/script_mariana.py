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
                              time_step = 2 * 3600, # 2 hr in seconds
                              uncertain = False,
                              )
    
    mapfile = os.path.join( base_dir, './mariana_island.bna')
    print "adding the map"
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=6*3600, #seconds
                                     )
    
    renderer = gnome.renderer.Renderer(mapfile, images_dir, size=(800, 600))
    #renderer.viewport = ((-76.5, 37.25),(-75.8, 37.75))
    
    print "adding outputters"
    model.outputters += renderer
    
    #netcdf_file = os.path.join(base_dir,'test_output.nc')
    #scripting.remove_netcdf(netcdf_file)
    #model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)
    
    print "adding a spill"
    
    # for now subsurface spill stays on initial layer - will need diffusion and rise velocity - wind doesn't act
    spill = gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (145.0, 15.0, 0.0),
                                            release_time = start_time,
                                            )
        
    model.spills += spill
    
    print  "adding a RandomMover:"
    r_mover = gnome.movers.RandomMover(diffusion_coef=10000)
    model.movers += r_mover
    
    
    print "adding a wind mover:"
    
    series = np.zeros((2,), dtype=gnome.basic_types.datetime_value_2d)
    # (time, (speed, direction) )
    series[0] = (start_time, ( 5,   265) )
    series[1] = (start_time+timedelta(hours=48), ( 5,      275) )
    
    
    wind = Wind(timeseries=series, units='m/s')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover
    
    # print "adding a current mover:"
    
    curr_file=os.path.join( base_dir, r"./HYCOM.nc")
    topology_file=r""
    #c_mover = gnome.movers.GridCurrentMover(curr_file, topology_file)
    c_mover = gnome.movers.GridCurrentMover(curr_file)
    model.movers += c_mover

    return model

if __name__ == "__main__":
    """ if called on its own -- run it """
    from gnome import scripting

    scripting.make_images_dir()
    model = make_model()
    model.full_run(log=True)
    


