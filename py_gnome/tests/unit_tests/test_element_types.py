'''
Unit test the classes in elementy_types module

Only contains tests for ArrayType since the remaining element_types classes
are trivial. They are tested in the integrated_tests
'''
import numpy as np
from gnome import basic_types
from gnome.movers.element_types import ArrayType

class TestArrayType_eq(object):
    """ 
    contains functions that test __eq__ for ArrayType object 
    """
    def test_eq_wrong_shape(self):
        """ array shape is different for two ArrayType objects """
        positions = ArrayType( (), basic_types.world_point_type)
        positions2= ArrayType( (3,), basic_types.world_point_type)
        assert positions != positions2      
    
    def test_eq_wrong_dtype(self):
        """ dtype is different for two ArrayType objects """
        positions = ArrayType( (3,), basic_types.world_point_type)
        positions2= ArrayType( (3,), np.int)
        assert positions != positions2      # wrong dtype
    
    def test_eq_wrong_init_value(self):
        """ initial_value is different for two ArrayType objects """
        status_codes = ArrayType( (), basic_types.status_code_type, basic_types.oil_status.in_water)
        status_codes2= ArrayType( (), basic_types.status_code_type)
        assert status_codes != status_codes2    # no init conditions
        
    def test_eq_wrong_attr(self):
        """ added an attribute so two ArrayType objects are diffferent """
        positions = ArrayType( (), basic_types.world_point_type)
        positions2= ArrayType( (3,), basic_types.world_point_type)
        positions2.test = 'test'
        assert positions != positions2      # wrong number of attributes
        
    def test_eq(self):
        """ both ArrayType objects are the same """
        positions = ArrayType( (3,), basic_types.world_point_type)
        positions2= ArrayType( (3,), basic_types.world_point_type)
        assert positions == positions2      # wrong shape

