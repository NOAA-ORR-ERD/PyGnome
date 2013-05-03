import json
from datetime import datetime,timedelta
import pprint
import os
import shutil

import numpy as np
import colander

import gnome
from gnome import movers
from gnome.persist import scenario

"""
Define a scenario and persist it to ./test_persist/
"""
saveloc  = './test_persist'
datafiles= os.path.join( os.path.dirname('__file__'),'../../scripts/script_boston')

if os.path.exists(saveloc):
    shutil.rmtree(saveloc)

os.mkdir(saveloc)

def make_model():
    mapfile = os.path.join( datafiles, './MassBayMap.bna')
    
    start_time = datetime(2013, 2, 13, 9, 0)
    model = gnome.model.Model(start_time = start_time,
                            duration = timedelta(days=2),
                            time_step = 30 * 60, # 1/2 hr in seconds
                            uncertain = True,
                            map= gnome.map.MapFromBNA(mapfile,
                                                      refloat_halflife=1*3600, #seconds
                                                      )
                            )
    
    print "adding a renderer"
    model.outputters += gnome.renderer.Renderer(mapfile,
                                                images_dir=os.path.join(datafiles,'images'),
                                                size=(800, 600))
    
    print "adding a spill"
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (144.664166, 13.441944, 0.0),
                                            release_time = start_time,
                                            end_release_time = start_time + timedelta(hours=6)
                                            )
    
    #need a scenario for SimpleMover
    #model.movers += movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    
    print  "adding a RandomMover:"
    model.movers += gnome.movers.RandomMover(diffusion_coef=100000)
    
    print "adding a wind mover:"
    
    series = np.zeros((2,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time,                      ( 5,   180) )
    series[1] = (start_time+timedelta(hours=18),  ( 5,   180) )
    
    w_mover = gnome.movers.WindMover( gnome.environment.Wind(timeseries=series,units='m/s') )
    model.movers += w_mover
    model.environment += w_mover.wind
    
    print "adding a cats shio mover:"
    
    c_mover = gnome.movers.CatsMover(os.path.join(datafiles,"./EbbTides.CUR"), 
                                     tide=gnome.environment.Tide(os.path.join( datafiles, "./EbbTidesShio.txt")))
    c_mover.scale_refpoint = (-70.8875, 42.321333) # this is the value in the file (default)
    c_mover.scale = True #default value
    c_mover.scale_value = -1 
    model.movers += c_mover
    model.environment += c_mover.tide    # todo: cannot add this till environment base class is created
    
    print "adding a cats ossm mover:"
    
    c_mover = gnome.movers.CatsMover(os.path.join(datafiles, "./MerrimackMassCoast.CUR"), 
                                     tide=gnome.environment.Tide(os.path.join(datafiles,"./MerrimackMassCoastOSSM.txt")) )
    c_mover.scale = True    # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.65,42.58333)
    c_mover.scale_value = 1.    
    model.movers += c_mover
    model.environment += c_mover.tide
    
    print "adding a cats mover:"
       
    c_mover = gnome.movers.CatsMover(os.path.join(datafiles,"MassBaySewage.CUR"))
    c_mover.scale = True    # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.78333,42.39333)
    c_mover.scale_value = .04    #the scale factor is 0 if user inputs no sewage outfall effects 
    model.movers += c_mover
    return model

def test_save_load_scenario():
    model = make_model()
    print "saving scnario .."
    scene = scenario.Scenario(saveloc, model)
    scene.save()
    scene.model = None   # make it none - load from persistence
    print "loading scenario .."
    model2 = scene.load()
    #model2 = scene.model
    
    assert model == model2