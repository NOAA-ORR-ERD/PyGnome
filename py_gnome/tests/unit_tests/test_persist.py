import json
from datetime import datetime,timedelta
import pprint
import os
import shutil

import numpy as np
import colander
import pytest

import gnome
from gnome import movers
from gnome.persist import scenario
from gnome.utilities.remote_data import get_datafile

"""
Define a scenario and persist it to ./test_persist/
"""
datafiles= os.path.join( os.path.dirname(__file__),'sample_data','boston_data')

@pytest.fixture(scope="function")
def setup_dirs(request):
    saveloc_  = os.path.join( os.path.dirname(__file__),'save_model')
    if os.path.exists(saveloc_):
        shutil.rmtree(saveloc_)
    os.makedirs(saveloc_)
    
    # create / clean up images dir as well
    images_dir = os.path.join(datafiles,'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        
    os.makedirs(images_dir)
    
    # Not sure we want default to delete the save_model dir 
    # maybe good to leave as an example
    def cleanup():
       print ("cleanup ..")
       if os.path.exists(saveloc_):
           shutil.rmtree(saveloc_)
           print ("shutil.rmtree({0})".format(saveloc_))
     
    request.addfinalizer(cleanup)
    return {'saveloc':saveloc_, 'images_dir':images_dir}

def make_model(images_dir, uncertain=False):
    mapfile = get_datafile( os.path.join( datafiles, './MassBayMap.bna') )
    
    start_time = datetime(2013, 2, 13, 9, 0)
    model = gnome.model.Model(start_time = start_time,
                            duration = timedelta(days=2),
                            time_step = 30 * 60, # 1/2 hr in seconds
                            uncertain = uncertain,
                            map= gnome.map.MapFromBNA(mapfile,
                                                      refloat_halflife=1, #hours
                                                      )
                            )
    
    print "adding a renderer"
    
    model.outputters += gnome.renderer.Renderer(mapfile,
                                                images_dir,
                                                size=(800, 600))
    
    print "adding a spill"
    model.spills += gnome.spill.PointSourceSurfaceRelease(num_elements=1000,
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
    
    d_file1 = get_datafile( os.path.join(datafiles,"./EbbTides.cur") )
    d_file2 = get_datafile( os.path.join( datafiles, "./EbbTidesShio.txt"))
    c_mover = gnome.movers.CatsMover( d_file1, 
                                      tide=gnome.environment.Tide( d_file2))
    c_mover.scale_refpoint = (-70.8875, 42.321333) # this is the value in the file (default)
    c_mover.scale = True #default value
    c_mover.scale_value = -1 
    model.movers += c_mover
    model.environment += c_mover.tide    # todo: cannot add this till environment base class is created
    
    print "adding a cats ossm mover:"
    
    d_file1 = get_datafile( os.path.join(datafiles,"./MerrimackMassCoast.cur") )
    d_file2 = get_datafile( os.path.join( datafiles, "./MerrimackMassCoastOSSM.txt"))
    c_mover = gnome.movers.CatsMover( d_file1, 
                                      tide=gnome.environment.Tide( d_file2 ) )
    c_mover.scale = True    # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.65,42.58333)
    c_mover.scale_value = 1.    
    model.movers += c_mover
    model.environment += c_mover.tide
    
    print "adding a cats mover:"
    
    d_file1 = get_datafile( os.path.join(datafiles,"MassBaySewage.cur") )   
    c_mover = gnome.movers.CatsMover( d_file1 )
    c_mover.scale = True    # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.78333,42.39333)
    c_mover.scale_value = .04    #the scale factor is 0 if user inputs no sewage outfall effects 
    model.movers += c_mover
    return model

@pytest.mark.parametrize("uncertain",[False, True])
def test_save_load_scenario(setup_dirs, uncertain):
    model = make_model(setup_dirs['images_dir'], uncertain)

    print "saving scnario .."
    scene = scenario.Scenario(setup_dirs['saveloc'], model)
    scene.save()
    
    scene.model = None   # make it none - load from persistence
    print "loading scenario .."
    model2 = scene.load()
    
    assert model == model2

@pytest.mark.parametrize("uncertain",[False, True])
def test_save_load_midrun_scenario(setup_dirs, uncertain):
    """
    create model, save it after 1step, then load and check equality of original model and persisted model
    """
    model = make_model(setup_dirs['images_dir'], uncertain)
    
    model.step()
    print "saving scnario .."
    scene = scenario.Scenario(setup_dirs['saveloc'], model)
    scene.save()
    
    scene.model = None   # make it none - load from persistence
    print "loading scenario .."
    model2 = scene.load()
    
    for sc in zip(model.spills.items(), model2.spills.items()):
        sc[0]._array_allclose_atol = 1e-5 # need to change both atol
        sc[1]._array_allclose_atol = 1e-5

    assert model.spills == model2.spills
    assert model == model2

@pytest.mark.parametrize("uncertain",[False, True])
def test_save_load_midrun_no_movers(setup_dirs, uncertain):
    """
    create model, save it after 1step, then load and check equality of original model and persisted model
    Remove all movers and ensure it still works as expected
    """
    model = make_model(setup_dirs['images_dir'], uncertain)
    
    for mover in model.movers:
        del model.movers[mover.id]
    
    model.step()
    print "saving scnario .."
    scene = scenario.Scenario(setup_dirs['saveloc'], model)
    scene.save()
    
    scene.model = None   # make it none - load from persistence
    print "loading scenario .."
    model2 = scene.load()
        
    for sc in zip(model.spills.items(), model2.spills.items()):
        sc[0]._array_allclose_atol = 1e-5 # need to change both atol since reading persisted data
        sc[1]._array_allclose_atol = 1e-5
    
    assert model.spills == model2.spills
    assert model == model2


@pytest.mark.parametrize("uncertain",[False, True])
def test_load_midrun_ne_rewound_model(setup_dirs, uncertain):
    """
    Load the same model that was persisted previously after 1 step
    This time rewind the original model and test that the two are not equal.
    The data arrays in the spill container must not match
    """
    # data arrays in model.spills no longer equal
    model = make_model(setup_dirs['images_dir'], uncertain)
    
    model.step()
    print "saving scnario .."
    scene = scenario.Scenario(setup_dirs['saveloc'], model)
    scene.save()
    
    model.rewind()  
    model2 = scene.load()
    
    assert model.spills != model2.spills
    assert model != model2 
