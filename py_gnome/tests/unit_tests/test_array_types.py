'''
Unit test the classes in elementy_types module

Only contains tests for ArrayType since the remaining array_types classes
are trivial. They are tested in the integrated_tests
'''

import numpy as np

from gnome.basic_types import world_point_type, oil_status, \
    status_code_type

from gnome.array_types import ArrayType, IdArrayType, ArrayTypeDivideOnSplit
from gnome.array_types import gat, reset_to_defaults
from pytest import mark, raises

from testfixtures import log_capture


age = gat('age')
mass = gat('mass')
mass_components = gat('mass_components')

def test_reset_to_defaults():
    '''
    will reset the object that found in array_types._default_values and
    ignore the one not found in this dict.
    '''
    ival = age.initial_value
    age.initial_value = 100
    reset_to_defaults(age)
    assert age.initial_value == ival


def test_get_array_type():
    '''
    will reset the object that found in array_types._default_values and
    ignore the one not found in this dict.
    '''
    id = gat('id')
    assert isinstance(id, IdArrayType)
    assert isinstance(age, ArrayType)
    assert isinstance(mass, ArrayTypeDivideOnSplit)


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
        positions2 = ArrayType((3, ), np.int32, 'positions')
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

#
# class TestSplitElement:
#     @log_capture()
#     def test_exception_split_gt_1(self, l):
#         with raises(ValueError):
#             age.split_element(1, 5)
#
#         l.check(('gnome.array_types.ArrayType',
#                  'ERROR',
#                  "'num' to split into must be at least 2"))
#
#     @log_capture()
#     def test_exception_l_frac_len(self, l):
#         with raises(ValueError):
#             mass.split_element(3, 5, l_frac=(.2, ))
#
#         l.check(('gnome.array_types.ArrayTypeDivideOnSplit',
#                  'ERROR',
#                  "in split_element() len(l_frac) must equal 'num'"))
#
#     @log_capture()
#     def test_exception_sum_l_frac_1(self, l):
#         with raises(ValueError):
#             mass.split_element(3, 5, l_frac=(.2, .1, .2))
#
#         l.check(('gnome.array_types.ArrayTypeDivideOnSplit',
#                  'ERROR',
#                  "sum 'l_frac' must be 1.0"))
#
#     # -------- END EXCEPTION TESTING
#     def _replace_le_after_split(self, vals, ix, split_elems):
#         '''
#         helper function - this is how spill_container will integrate new elements
#         into original array
#         '''
#         # insert split_elems until the last element before ix, then replace
#         # ix with last element of split_elems. This doesn't require a delete, then
#         # insert. It also works for splittle all elements.
#         new_vals = np.insert(vals, ix, split_elems[:-1], 0)
#         new_vals[ix + len(split_elems) - 1] = split_elems[-1]
#         return new_vals
#
#     @mark.parametrize("num", (2, 4))
#     def test_split_element_clone_values(self, num):
#         ''' as an example age should get cloned upon split '''
#         # use an object of ArrayType (base) instance
#         vals = age.initialize(10)
#         vals[:] = np.arange(0, 100, 10)
#         split_into = num
#         ix = 5
#         split_elems = age.split_element(split_into, vals[ix])
#         assert split_elems.dtype == vals.dtype
#         assert np.all(split_elems[:] == vals[ix])
#         assert len(split_elems) == split_into
#         new_vals = self._replace_le_after_split(vals, ix, split_elems)
#
#         assert np.all(new_vals[ix:ix + split_into] == new_vals[ix])
#         assert np.all(new_vals[:ix] == vals[:ix])
#         assert np.all(new_vals[ix + split_into:] == vals[ix + 1:])
#         assert len(new_vals) == len(vals) + num - 1
#
#     @mark.parametrize(("num", "l_frac"), [(3, None),
#                                           (2, (.6, .4)),
#                                           (3, (.6, .3, .1))])
#     def test_split_element_divide(self, num, l_frac):
#         '''
#         test mass/mass_components being split correctly
#         '''
#         num_les = 5
#         m_array = mass.initialize(num_les)
#         m_array[:] = 10
#         mc = mass_components.initialize(num_les,
#                                         (4,),
#                                         np.asarray([0.2, 0.2, 0.4, 0.2])*10)
#         # split element 3 into 2 elements
#         ix = 3
#         m_split = mass.split_element(num, m_array[ix], l_frac)
#         new_m_array = self._replace_le_after_split(m_array, ix, m_split)
#
#         mc_split = mass_components.split_element(num, mc[ix], l_frac)
#         if l_frac is None:
#             assert np.all(mc_split[0] == mc_split)
#         else:
#             for i in xrange(len(mc_split)):
#                 # check l_frac is correctly applied
#                 assert np.allclose(mc_split[i], l_frac[i] * mc[ix])
#
#         assert np.allclose(mc_split.sum(0), mc[ix])
#
#         new_mc_array = self._replace_le_after_split(mc, ix, mc_split)
#         assert np.allclose(new_mc_array.sum(1), new_m_array)
