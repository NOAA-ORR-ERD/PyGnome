"""
Tests for cy_shio_time.pyx class

Does some basic testing of properties. 

todo: still need to add tests for functions that get the data from C++ code
"""
import os

import pytest

import gnome
from gnome.cy_gnome.cy_shio_time import CyShioTime

shio_file = os.path.join( os.path.dirname(__file__), r"sample_data","tides","CLISShio.txt")

def test_exceptions():
    bad_file = os.path.join(r"sample_data","long_island_sound","CLISShio.txtX")
    bad_yeardata_path = os.path.join(r"sample_data","Data","yeardata")
    with pytest.raises(IOError):
        CyShioTime(bad_file)
        
    with pytest.raises(IOError):
        shio = CyShioTime(shio_file)
        shio.set_shio_yeardata_path(bad_yeardata_path)
        

        
check_items={'Type=':'station_type','Name=':'station','Longitude=':'station_location','Latitude=':'station_location'}

def test_properties_read_from_file():
    """ read only properties """
    shio = CyShioTime(shio_file)
    with open(shio_file) as fh:
        data = fh.read()
        
    for item, prop in check_items.iteritems():
        s_idx = data.find( item )
        e_idx = data.find( '\n', s_idx)
        val = getattr( shio, prop)
        if item == 'Longitude=':
            assert round( val[0], 4 )  == round( float( data[s_idx+len(item):e_idx]), 4)
            
        elif item == 'Latitude=':
            assert round( val[1], 4 ) == round( float( data[s_idx+len(item):e_idx]), 4)
            
        else: 
            assert val == data[s_idx+len(item):e_idx]
            
    assert shio.daylight_savings_off
    assert shio.filename == shio_file
            
def test_daylight_savings_off():
    shio = CyShioTime(shio_file)
    assert shio.daylight_savings_off
    shio.daylight_savings_off = False
    assert not shio.daylight_savings_off
    
def test_scale_factor():
    shio = CyShioTime(shio_file)
    assert shio.scale_factor == 1
    shio.scale_factor = 2
    assert shio.scale_factor == 2
    
def test_yeardata():
    shio = CyShioTime(shio_file)
    assert shio.yeardata == ''
    yd = os.path.join( os.path.dirname( gnome.__file__), 'data','yeardata/')
    shio.yeardata = yd
    assert shio.yeardata == yd
    
