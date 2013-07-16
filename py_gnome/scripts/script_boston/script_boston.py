#!/usr/bin/env python

"""
a simple script to run GNOME

This one uses:

  - the GeoProjection
  - wind mover
  - random mover
  - cats shio mover
  - cats ossm mover
  - plain cats mover 

"""


import os
import shutil
from datetime import datetime, timedelta
import argparse
import sys

import numpy as np

import gnome
from gnome.environment import Wind, Tide
from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome.persist import scenario
from gnome import scripting

# define base directory
base_dir = os.path.dirname(__file__)

def make_model(images_dir=os.path.join(base_dir,"images")):
    # create the maps:
    print "creating the maps"
    
    mapfile = os.path.join( base_dir, './MassBayMap.bna')
    gnome_map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=1, #hours
                                     )
    
    renderer = gnome.renderer.Renderer(mapfile,
                                       images_dir,
                                       size=(800, 800),
                                       projection_class=gnome.utilities.projections.GeoProjection)
    
    
    print "initializing the model"
    
    start_time = datetime(2013, 3, 12, 10, 0)
    model = gnome.model.Model(time_step=900, # 15 minutes in seconds
                              start_time=start_time, # default to now, rounded to the nearest hour
                              duration=timedelta(days=1),
                              map=gnome_map,
                              uncertain=False,)
    
    print "adding outputters"
    model.outputters += renderer
    
    netcdf_file = os.path.join(base_dir,'script_boston.nc')
    scripting.remove_netcdf(netcdf_file)
    model.outputters += gnome.netcdf_outputter.NetCDFOutput(netcdf_file, all_data=True)
    
    print  "adding a RandomMover:"
    model.movers += gnome.movers.RandomMover(diffusion_coef=100000)
    
    
    print "adding a wind mover:"
    
    series = np.zeros((2,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time,                      ( 5,   180) )
    series[1] = (start_time+timedelta(hours=18),  ( 5,   180) )
    
    
    w_mover = gnome.movers.WindMover( Wind(timeseries=series,units='m/s') )
    model.movers += w_mover
    model.environment += w_mover.wind
    
    print "adding a cats shio mover:"
    
    curr_file=os.path.join( base_dir, r"./EbbTides.CUR")
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(os.path.join( base_dir, r"./EbbTidesShio.txt")))
    c_mover.scale_refpoint = (-70.8875, 42.321333) # this is the value in the file (default)
    c_mover.scale = True #default value
    c_mover.scale_value = -1 
    model.movers += c_mover
    model.environment += c_mover.tide    # todo: cannot add this till environment base class is created
    
    print "adding a cats ossm mover:"
    
    ossm_file = os.path.join( base_dir, r"./MerrimackMassCoastOSSM.txt")
    curr_file=os.path.join( base_dir, r"./MerrimackMassCoast.CUR")
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(os.path.join( base_dir, "./MerrimackMassCoastOSSM.txt")))
    # but do need to scale (based on river stage)
    c_mover.scale = True
    c_mover.scale_refpoint = (-70.65,42.58333)
    c_mover.scale_value = 1.    
    model.movers += c_mover
    model.environment += c_mover.tide
    
    print "adding a cats mover:"
    
    curr_file=os.path.join( base_dir, r"MassBaySewage.CUR")
    c_mover = gnome.movers.CatsMover(curr_file)
    # but do need to scale (based on river stage)
    c_mover.scale = True
    c_mover.scale_refpoint = (-70.78333,42.39333)
    c_mover.scale_value = .04    #the scale factor is 0 if user inputs no sewage outfall effects 
    model.movers += c_mover
    
    # print "adding a component mover:"
    # component_file1 =  os.path.join( base_dir, r"./WAC10msNW.cur")
    # component_file2 =  os.path.join( base_dir, r"./WAC10msSW.cur")
    
    print "adding a spill"
    
    spill = gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (-70.911432, 42.369142, 0.0),
                                            release_time = start_time,
                                            end_release_time = start_time + timedelta(hours=12),
                                            )
    
    model.spills += spill
    
    return model