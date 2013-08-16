'''
Tests serialize/deserialize to create and update objects

It just tests the interface works, doesn't actually change values
'''
import os

import pytest

from gnome import environment, movers
from gnome.persist import environment_schema as env_schema
from gnome.persist import movers_schema
from gnome.utilities.remote_data import get_datafile

here = os.path.dirname(__file__)
data_dir = os.path.join(here, 'sample_data')
tides_dir = os.path.join(data_dir, 'tides')
lis_dir = os.path.join(data_dir, 'long_island_sound')


def test_wind_create(wind_circ):
    """
    wind_circ is a fixture
    """
    c_dict = env_schema.Wind().serialize(wind_circ['wind'].to_dict('create'))
    dict_ = env_schema.Wind().deserialize(c_dict)

    # now use dict to create new object
    new_w = environment.Wind.new_from_dict(dict_)
    assert new_w == wind_circ['wind']


def test_wind_update(wind_circ):
    """
    Just tests methods don't fail and the schema is properly defined.
    It doesn't update any properties.

    Separate/independent test for serializable class

    wind_circ is a fixture
    """
    c_dict = env_schema.Wind().serialize(wind_circ['wind'].to_dict())
    dict_ = env_schema.Wind().deserialize(c_dict)
    wind_circ['wind'].from_dict(dict_)
    assert True 
    
    
@pytest.mark.parametrize("filename", [get_datafile( os.path.join( tides_dir, "CLISShio.txt") ), 
                                      get_datafile( os.path.join( tides_dir, r"TideHdr.FINAL") )])
def test_tide_create(filename):
    td = environment.Tide(filename=filename)
    c_dict = env_schema.Tide().serialize(td.to_dict('create'))
    dict_ = env_schema.Tide().deserialize(c_dict)

    # now use dict to create new object
    new_w = environment.Tide.new_from_dict(dict_)
    assert new_w == td
    
    
@pytest.mark.parametrize("filename", [get_datafile( tides_dir, "CLISShio.txt") ), 
                                      get_datafile( tides_dir, "TideHdr.FINAL") )])
def test_tide_update(filename):
    """
    Just tests methods don't fail and the schema is properly defined.
    It doesn't update any properties.

    Separate/independent test for serializable class

    wind_circ is a fixture
    """
    td = environment.Tide(filename=filename)
    c_dict = env_schema.Tide().serialize(td.to_dict())
    dict_ = env_schema.Tide().deserialize(c_dict)
    td.from_dict(dict_)
    assert True


def test_windmover_create(wind_circ):
    wm = movers.WindMover(wind_circ['wind'])
    c_dict = movers_schema.WindMover().serialize(wm.to_dict('create'))
    dict_ = movers_schema.WindMover().deserialize(c_dict)

    # need to include wind object associated with wind_id
    dict_.update({'wind': wind_circ['wind']})

    new_w = movers.WindMover.new_from_dict(dict_)
    assert wm == new_w


def test_windmover_update(wind_circ):
    wm = movers.WindMover(wind_circ['wind'])
    c_dict = movers_schema.WindMover().serialize(wm.to_dict())
    dict_ = movers_schema.WindMover().deserialize(c_dict)

    # now let's say we want to update the Wind object
    wind = environment.Wind(timeseries=wind_circ['rq'], units='m/s')
    dict_.update({'wind': wind})
    wm.from_dict(dict_)
    assert wm.wind.id == wind.id


def test_catsmover_update():
    curr_file= get_datafile( os.path.join( lis_dir, "tidesWAC.CUR") )
    td_file  = get_datafile( os.path.join( lis_dir, "CLISShio.txt") )
    c_mv = movers.CatsMover(curr_file, tide=environment.Tide(td_file) )
    c_dict = movers_schema.CatsMover().serialize( c_mv.to_dict() )
    dict_ = movers_schema.CatsMover().deserialize( c_dict)
    
    # now let's say we want to update the Tide object, which is not part of the serialization
    tide = environment.Tide(td_file)
    dict_.update({'tide': tide})
    c_mv.from_dict(dict_)
    assert c_mv.tide.id == tide.id
