'''
Test various element types available for the Spills
'''
import copy
import numpy as np
import pytest


from gnome.array_types import (windages, windage_range, windage_persist,
                               mass,
                               rise_vel)
from gnome.elements import (InitWindagesConstantParams,
                            InitMassFromVolume,
                            InitRiseVelFromDist)
from gnome.spill import Spill

from conftest import mock_append_data_arrays


""" Helper functions """
windages = {'windages': windages,
            'windage_range': windage_range,
            'windage_persist': windage_persist}
mass_array = {'mass': mass}
rise_vel_array = {'rise_vel': rise_vel}
num_elems = 10


def assert_dataarray_shape_size(arr_types, data_arrays, num_released):
    for key, val in arr_types.iteritems():
        assert data_arrays[key].dtype == val.dtype
        assert data_arrays[key].shape == (num_released,) + val.shape


""" Initializers """


@pytest.mark.parametrize(("fcn", "arr_types", "spill"),
                [(InitWindagesConstantParams(), windages, None),
                 (InitWindagesConstantParams(), windages, None),
                 (InitMassFromVolume(), mass_array, Spill(volume=10)),
                 (InitRiseVelFromDist(), rise_vel_array, None),
                 (InitRiseVelFromDist('normal'), rise_vel_array, None),
                 ])
def test_correct_particles_set_by_initializers(fcn, arr_types, spill):
    """
    Tests that the correct elements (ones that
    were released last) are initialized
    """
    # let's only set the values for the last 10 elements
    # this is not how it would be used, but this is just to make sure
    # the values for the correct elements are set
    data_arrays = mock_append_data_arrays(arr_types, num_elems)
    data_arrays = mock_append_data_arrays(arr_types, num_elems, data_arrays)

    fcn.initialize(num_elems, spill, data_arrays)

    assert_dataarray_shape_size(arr_types, data_arrays, num_elems * 2)

    # contrived example since particles will be initialized for every timestep
    # when they are released. But just to make sure that only values for the
    # latest released elements are set
    for key in data_arrays:
        assert np.all(0 == data_arrays[key][:num_elems])

        # values for these particles should be initialized to non-zero
        assert np.any(0 != data_arrays[key][-num_elems:])


class TestInitConstantWindageRange:
    @pytest.mark.parametrize(("fcn", "array"),
            [(InitWindagesConstantParams(), windages),
             (InitWindagesConstantParams([0.02, 0.03]), windages),
             (InitWindagesConstantParams(), windages),
             (InitWindagesConstantParams(windage_persist=-1), windages)])
    def test_initailize_InitConstantWindageRange(self, fcn, array):
        """
        tests initialize method
        """
        data_arrays = mock_append_data_arrays(array, num_elems)
        fcn.initialize(num_elems, None, data_arrays)
        assert_dataarray_shape_size(array, data_arrays, num_elems)

        assert np.all(data_arrays['windage_range'] == fcn.windage_range)
        assert np.all(data_arrays['windage_persist'] == fcn.windage_persist)
        np.all(data_arrays['windages'] != 0)
        np.all(data_arrays['windages'] >= data_arrays['windage_range'][:, 0])
        np.all(data_arrays['windages'] <= data_arrays['windage_range'][:, 1])

    def test_exceptions(self):
        bad_wr = [-1, 0]
        bad_wp = 0
        obj = InitWindagesConstantParams()
        with pytest.raises(ValueError):
            InitWindagesConstantParams(windage_range=bad_wr)

        with pytest.raises(ValueError):
            InitWindagesConstantParams(windage_persist=bad_wp)

        with pytest.raises(ValueError):
            obj.windage_range = bad_wr

        with pytest.raises(ValueError):
            obj.windage_persist = bad_wp


def test_initailize_InitMassFromVolume():
    data_arrays = mock_append_data_arrays(mass_array, num_elems)
    fcn = InitMassFromVolume()
    spill = Spill()
    spill.volume = num_elems / (spill.oil_props.get_density('kg/m^3') * 1000)
    fcn.initialize(num_elems, spill, data_arrays)

    assert_dataarray_shape_size(mass_array, data_arrays, num_elems)
    assert np.all(1. == data_arrays['mass'])


def test_initialize_InitRiseVelFromDist_uniform():
    """
    test initialize data_arrays with uniform dist
    """
    data_arrays = mock_append_data_arrays(rise_vel_array, num_elems)
    fcn = InitRiseVelFromDist()
    fcn.initialize(num_elems, None, data_arrays)

    assert_dataarray_shape_size(rise_vel_array, data_arrays, num_elems)

    assert np.all(0 != data_arrays['rise_vel'])
    assert np.all(data_arrays['rise_vel'] <= 1)
    assert np.all(data_arrays['rise_vel'] >= 0)


def test_initialize_InitRiseVelFromDist_normal():
    """
    test initialize data_arrays with normal dist
    assume normal distribution works fine - so statistics (mean, var) are not
    tested
    """
    num_elems = 1000
    data_arrays = mock_append_data_arrays(rise_vel_array, num_elems)
    fcn = InitRiseVelFromDist('normal')
    fcn.initialize(num_elems, None, data_arrays)

    assert_dataarray_shape_size(rise_vel_array, data_arrays, num_elems)

    assert np.all(0 != data_arrays['rise_vel'])


""" Element Types """
