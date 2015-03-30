'''
Unit test the classes in elementy_types module

Only contains tests for ArrayType since the remaining array_types classes
are trivial. They are tested in the integrated_tests
'''

import numpy as np

from gnome.basic_types import world_point_type, oil_status, \
    status_code_type

from gnome.array_types import ArrayType


class TestArrayType_eq(object):

    """
    contains functions that test __eq__ for ArrayType object
    """

    def test_eq_wrong_shape(self):
        """ array shape is different for two ArrayType objects """

        positions = ArrayType((), world_point_type, 'positions')
        positions2 = ArrayType((3, ), world_point_type, 'positions')
        assert positions != positions2

    def test_eq_wrong_dtype(self):
        """ dtype is different for two ArrayType objects """

        positions = ArrayType((3, ), world_point_type, 'positions')
        positions2 = ArrayType((3, ), np.int, 'positions')
        assert positions != positions2  # wrong dtype

    def test_eq_wrong_init_value(self):
        """ initial_value is different for two ArrayType objects """

        status_codes = ArrayType((), status_code_type, 'name',
                                 oil_status.in_water)
        status_codes2 = ArrayType((), status_code_type, 'name')
        assert status_codes != status_codes2  # no init conditions

    def test_eq_wrong_attr(self):
        """ added an attribute so two ArrayType objects are diffferent """

        positions = ArrayType((), world_point_type, 'name')
        positions2 = ArrayType((3, ), world_point_type, 'name')
        positions2.test = 'test'
        assert positions != positions2  # wrong number of attributes

    def test_eq_wrong_name(self):
        """ added an attribute so two ArrayType objects are diffferent """

        positions = ArrayType((3, ), world_point_type, 'positions')
        positions2 = ArrayType((3, ), world_point_type, 'positions2')
        assert positions != positions2  # wrong number of attributes

    def test_eq(self):
        """ both ArrayType objects are the same """

        positions = ArrayType((3, ), world_point_type, 'positions')
        positions2 = ArrayType((3, ), world_point_type, 'positions')
        assert positions == positions2  # wrong shape
