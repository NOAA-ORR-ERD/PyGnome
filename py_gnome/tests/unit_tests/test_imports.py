"""
basic test to see if everything imports successfully
"""

# import the pure python modules:

def test_import_gnome():
    import gnome
    
def test_import_map():
    import gnome.map

#def test_import_model():
#    import gnome.model

def test_import_spill():
    import gnome.spill

def test_import_greenwich():
    import gnome.greenwich

## import the cython extensions:

def test_import_basic_types():
    import gnome.cy_basic_types

def test_import_netcdf_mover():
    import gnome.cy_netcdf_mover

def test_import_ossm_time():
    import gnome.cy_ossm_time

#def test_import_cats_mover():
#    import gnome.cy_cats_mover
#    
#def test_import_random_mover():
#    import gnome.cy_random_mover

def test_import_wind_mover():
    import gnome.cy_wind_mover
    
    

