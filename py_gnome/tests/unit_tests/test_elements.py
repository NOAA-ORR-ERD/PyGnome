'''
Test various element types available for the Spills
'''
import copy
import numpy as np
import pytest

from gnome.element_type import SubsurfaceRiseVelDist
from gnome.array_types import RiseVelocityMover

from conftest import mock_append_data_arrays


class TestSubsurfaceRiseVelDist:
    """ """
    arr_types = dict(RiseVelocityMover)
    s = SubsurfaceRiseVelDist()

    def setup_helper(self, num_elems=0, dist='uniform', params=[0, 1],
                     d_arrays={}):
        """ put all the common setup calls for tests here"""
        self.s.distribution = dist
        self.s.params = params
        self.data_arrays = mock_append_data_arrays(self.arr_types, num_elems,
                                                   d_arrays)
        self.zero_rise_vel = np.copy(self.data_arrays['rise_vel'])

    def test_init(self):
        """
        raise init error
        """
        with pytest.raises(ValueError):
            SubsurfaceRiseVelDist(distribution='binomial')

        assert self.s.distribution == 'uniform'
        assert self.s.params[0] == 0 and self.s.params[1] == 1

    def test_setup_helper(self):
        self.setup_helper(10)
        assert 'rise_vel' in self.data_arrays
        assert np.all(self.zero_rise_vel == 0)

    def test_set_newparticle_values_with_0_elements(self):
        """ nothing happens if 0 particles are released """
        self.setup_helper()
        assert len(self.zero_rise_vel) == 0
        self.s.set_newparticle_values(0, None, self.data_arrays)
        assert len(self.data_arrays['rise_vel']) == 0

    def test_uniform_set_newparticle_values_with_dataarray(self):
        """
        test setting up data_arrays
        """
        num_elems = 10
        self.setup_helper(num_elems)
        self.s.set_newparticle_values(num_elems, None, self.data_arrays)
        assert len(self.data_arrays['rise_vel'] == num_elems)
        assert (self.arr_types['rise_vel'].dtype ==
                        self.data_arrays['rise_vel'].dtype)
        assert (self.zero_rise_vel.shape ==
                        self.data_arrays['rise_vel'].shape)
        assert np.all(self.zero_rise_vel !=
                        self.data_arrays['rise_vel'])
        assert np.all(self.data_arrays['rise_vel'] <= 1)
        assert np.all(self.data_arrays['rise_vel'] >= 0)

    def test_normal_set_newparticle_values_with_dataarray(self):
        """
        test setting up data_arrays
        """
        num_elems = 1000
        self.setup_helper(num_elems)
        # spill is not used for this function so 'None' is given as input
        self.s.set_newparticle_values(num_elems, None, self.data_arrays)
        assert len(self.data_arrays['rise_vel'] == num_elems)
        assert (self.arr_types['rise_vel'].dtype ==
                        self.data_arrays['rise_vel'].dtype)
        assert (self.zero_rise_vel.shape ==
                        self.data_arrays['rise_vel'].shape)
        assert np.all(self.zero_rise_vel
                       != self.data_arrays['rise_vel'])

    def test_set_newparticle_values_append_dataarray(self):
        """
        test append to data_arrays
        test that set_newparticle_values() only sets values for the newly
        released elements
        """
        num_elems = 5
        self.setup_helper(num_elems)    # let's leave these as 0
        self.setup_helper(num_elems, d_arrays=self.data_arrays)
        assert len(self.data_arrays['rise_vel']) == 2 * num_elems

        self.s.set_newparticle_values(num_elems, None, self.data_arrays)
        assert np.all(self.zero_rise_vel[:num_elems] ==
                      self.data_arrays['rise_vel'][:num_elems])
        assert np.all(self.zero_rise_vel[num_elems:] !=
                      self.data_arrays['rise_vel'][num_elems:])
