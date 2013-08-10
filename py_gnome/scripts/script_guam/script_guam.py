#!/usr/bin/env python

"""
Script to test GNOME with guam data

"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind, Tide

from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome import scripting

# define base directory
base_dir = os.path.dirname(__file__)

def make_model(images_dir=os.path.join(base_dir,"images")):
    print "initializing the model"
    
    start_time = datetime(2013, 2, 13, 9, 0)
    model = gnome.model.Model(start_time = start_time,
                              duration = timedelta(days=2),
                              time_step = 30 * 60, # 1/2 hr in seconds
                              uncertain = False,
                              )
    
    print "adding the map"
    
    mapfile = os.path.join( base_dir, './GuamMap.bna')
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=6, #hours
                                     )
    
    print "adding outputters"
    
    renderer = gnome.renderer.Renderer(mapfile,
                                       images_dir,
                                       size=(800, 600))
    renderer.viewport = ((144.6, 13.4),(144.7, 13.5)) 
    model.outputters += renderer
    
    netcdf_file = os.path.join(base_dir,'script_guam.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)
    
    print "adding a spill"
    
    spill = gnome.spill.PointSourceSurfaceRelease(num_elements=1000,
                                            start_position = (144.664166, 13.441944, 0.0),
                                            release_time = start_time,
                                            end_release_time = start_time + timedelta(hours=6)
                                            )
        
    model.spills += spill
    
    print  "adding a RandomMover:"
    r_mover = gnome.movers.RandomMover(diffusion_coef=50000)
    model.movers += r_mover
    
    
    print "adding a wind mover:"
    
    series = np.zeros((4,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, ( 5,   135) )
    series[1] = (start_time+timedelta(hours=23), ( 5,      135) )
    series[2] = (start_time+timedelta(hours=25), ( 5,     0) )
    series[3] = (start_time+timedelta(hours=48), ( 5,     0) )
    
    
    wind = Wind(timeseries=series,units='knot')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover
    model.environment += w_mover.wind
    
    print "adding a cats mover:"
    
    curr_file=os.path.join( base_dir, r"./OutsideWAC.cur")
    c_mover = gnome.movers.CatsMover(curr_file)
    c_mover.scale = True
    c_mover.scale_refpoint = (144.601, 13.42)
    c_mover.scale_value = .15
    model.movers += c_mover
    
    print "adding a cats shio mover:"
     
    curr_file=os.path.join( base_dir, r"./WACFloodTide.cur")
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(os.path.join( base_dir, r"./WACFTideShioHts.txt")))
    c_mover.scale_refpoint = (144.621667, 13.45) # this is different from the value in the file!
    c_mover.scale = True #default value
    c_mover.scale_value = 1 #default value
    c_mover.tide.scale_factor = 1.1864	#will need the fScaleFactor for heights files
    # #c_mover.time_dep.scale_factor = 1.1864	#will need the fScaleFactor for heights files
    model.movers += c_mover
    model.environment += c_mover.tide
 
    return model




