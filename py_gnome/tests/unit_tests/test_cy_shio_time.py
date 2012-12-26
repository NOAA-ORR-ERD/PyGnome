from gnome.cy_gnome import cy_shio_time
from datetime import datetime, timedelta
from gnome.utilities import time_utils
import numpy as np
import string
import pytest

def test_exceptions():
    with pytest.raises(IOError):
        """ Bad path - raise IOError """
        file=r"XXX/SampleData/long_island_sound/CLISShio.txt"
        cy_shio_time.CyShioTime(file)


file=r"SampleData/long_island_sound/CLISShio.txt"
shio = cy_shio_time.CyShioTime(file)
        
def test_properties():
    """
    test properties set correctly
    """
    assert shio.filename == file 
    assert shio.id == id(shio)
    assert shio.daylight_savings_off == True
    
    shio.daylight_savings_off = False
    assert shio.daylight_savings_off == False
    shio.daylight_savings_off = True
    
    
def test_str_and_repr():
    """
    prints the object 'shio' and calls repr(shio) to make sure they work
    """
    print
    print "-----------------------" 
    print "test_str_and_repr()"
    print 
    print shio
    print repr(shio)
    print "-----------------------"
    assert True
    
def test_correctness_of_data_read():
    """
    Let's read data from the file and make sure we get same info
    """
    file_ = open(file)
    data_ = file_.read()
    
    # check string info read correctly
    str_info = ['Name=','Type=','Latitude=','Longitude=']
    start_ix = [string.find(data_, "=", st)+1 for st in [string.find(data_, li) for li in str_info]]
    stop_ix  = [string.find(data_, "\n", st) for st in start_ix]
    info_    = [data_[start_ix[i]:stop_ix[i]] for i in range(len(str_info))]
    
    info  = shio.get_info()
    assert info['StationName'] == info_[0]
    assert info['StationType'] == info_[1]
    assert info['Lat'] == round(float(info_[2]), 2)
    assert info['Long'] == round(float(info_[3]), 2)



def test_get_time_value():
    """
    For now just tests the get_time_value method executes and returns non-zero results for u
    and zero for v in (u,v) pairs 
    """
    dt = [datetime.now() + timedelta(hours=(i*4)) for i in range(5)]
    dtsec   = time_utils.date_to_sec(dt)
    vel_rec = shio.get_time_value(dtsec)
    
    assert np.all(vel_rec['u'] != 0)    # TODO: check if this is always valid?
    assert np.all(vel_rec['v'] == 0)
    
