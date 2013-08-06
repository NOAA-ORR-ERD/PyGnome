'''
Tests serialize/deserialize to create and update objects

It just tests the interface works, doesn't actually change values
'''
import datetime
import time
import os
import json

import pytest

from gnome import environment, movers
from gnome.persist import environment_schema, movers_schema
from gnome.utilities.remote_data import get_datafile

def test_wind_create(wind_circ):
    """
    wind_circ is a fixture
    """
    c_dict = environment_schema.Wind().serialize( wind_circ['wind'].to_dict('create') )
    dict_ = environment_schema.Wind().deserialize(c_dict)
    new_w = environment.Wind.new_from_dict(dict_)  # now use dict to create new object
    assert new_w == wind_circ['wind']

def test_wind_update(wind_circ):
    """
    Just tests methods don't fail and the schema is properly defined. It doesn't update any properties
    
    Separate/independent test for serializable class
    
    wind_circ is a fixture
    """
    c_dict = environment_schema.Wind().serialize( wind_circ['wind'].to_dict() )
    dict_ = environment_schema.Wind().deserialize(c_dict)
    wind_circ['wind'].from_dict(dict_)
    assert True 
    
    
@pytest.mark.parametrize("filename", [get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/tides/CLISShio.txt") ), 
                                      get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/tides/TideHdr.FINAL") )])
def test_tide_create(filename):
    td = environment.Tide(filename=filename)
    c_dict = environment_schema.Tide().serialize( td.to_dict('create') )
    dict_ = environment_schema.Tide().deserialize(c_dict)
    new_w = environment.Tide.new_from_dict(dict_)  # now use dict to create new object
    assert new_w == td
    
    
@pytest.mark.parametrize("filename", [get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/tides/CLISShio.txt") ), 
                                      get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/tides/TideHdr.FINAL") )])
def test_tide_update(filename):
    """
    Just tests methods don't fail and the schema is properly defined. It doesn't update any properties
    
    Separate/independent test for serializable class
    
    wind_circ is a fixture
    """
    td = environment.Tide(filename=filename)
    c_dict = environment_schema.Tide().serialize( td.to_dict() )
    dict_ = environment_schema.Tide().deserialize(c_dict)
    td.from_dict(dict_)
    assert True 


def test_windmover_create(wind_circ):
    wm = movers.WindMover( wind_circ['wind'])
    c_dict = movers_schema.WindMover().serialize( wm.to_dict('create') )
    dict_ = movers_schema.WindMover().deserialize( c_dict)
    dict_.update({'wind':wind_circ['wind']})    # need to include wind object associated with wind_id
    new_w = movers.WindMover.new_from_dict( dict_ )
    assert wm == new_w

def test_windmover_update(wind_circ):
    wm = movers.WindMover( wind_circ['wind'])
    c_dict = movers_schema.WindMover().serialize( wm.to_dict() )
    dict_ = movers_schema.WindMover().deserialize( c_dict)
    
    # now let's say we want to update the Wind object
    wind = environment.Wind(timeseries=wind_circ['rq'], units='m/s')
    dict_.update({'wind':wind})
    wm.from_dict(dict_)
    assert wm.wind.id == wind.id


def test_catsmover_update():
    curr_file= get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/long_island_sound/tidesWAC.CUR") )
    td_file  = get_datafile( os.path.join( os.path.dirname(__file__), r"sample_data/long_island_sound/CLISShio.txt") )
    c_mv = movers.CatsMover(curr_file, tide=environment.Tide(td_file) )
    c_dict = movers_schema.CatsMover().serialize( c_mv.to_dict() )
    dict_ = movers_schema.CatsMover().deserialize( c_dict)
    
    # now let's say we want to update the Tide object, which is not part of the serialization
    tide = environment.Tide(td_file)
    dict_.update({'tide':tide})
    c_mv.from_dict(dict_)
    assert c_mv.tide.id == tide.id
