'''
Types of elements that a spill can expect
These are properties that are spill specific like:
  'floating' element_types would contain windage_range, windage_persist
  'subsurface_dist' element_types would contain rise velocity distribution info
  'nonweathering' element_types would set use_droplet_size flag to False
  'weathering' element_types would set use_droplet_size flag to True

  'SubsurfaceRiseVelDist' element_types would contain rise velocity
      distribution info
'''
from gnome import array_types
import numpy as np


class RiseVelDist_initializer(object):
    def __init__(self, distribution='uniform', params=[0, 1]):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        Use distribution to define rise_vel

        :param risevel_dist: could be 'uniform' or 'normal'
        :type distribution: str

        :param params: for 'uniform' dist, it is [min_val, max_val].
            For 'normal' dist, it is [mean, sigma] where sigma is
            1 standard deviation
        """

        if distribution not in ['uniform', 'normal']:
            raise ValueError("{0} is unknown distribution. Only 'uniform' or"
                             " 'normal' distribution is implemented")

        self.distribution = distribution
        self.params = params

        # should be only one key in dict, 'rise_vel'
        self.key = array_types.rise_vel.keys()[0]

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        """
        if 'rise_vel' exists in SpillContainer's data_arrays, then define
        """
        if self.key not in data_arrays or num_new_particles == 0:
            return

        if self.distribution == 'uniform':
            data_arrays[self.key][-num_new_particles:] = np.random.uniform(
                                                        self.params[0],
                                                        self.params[1],
                                                        num_new_particles)
        elif self.distribution == 'normal':
            data_arrays[self.key][-num_new_particles:] = np.random.normal(
                                                        self.params[0],
                                                        self.params[1],
                                                        num_new_particles)


class RiseVelWindageOil(object):
    def __init__(self):
        self.initializers = {'rise_vel': RiseVelDist_initializer}

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        #======================================================================
        # for key, value in self.initializers.iteritems():
        #     if key in data_arrays:
        #         self.initializers[key].set_newparticle_values(self,
        #                                 num_new_particles, spill, data_arrays)
        #======================================================================

        for key, val in data_arrays.iteritems():
            self.initializers[key].set_newparticle_values(self,
                                    num_new_particles, spill, val)

#==============================================================================
# class MassFromTotalVolume(object):
#     """
#     This could also go into ArrayType's initialize method if we pass
#     spill as input to initialize method
#     """
#     def __init__(self):
#         """
#         mass units are 'grams'
#         """
#         self.key = dict(array_types.mass).keys()
#  
#     def set_values(self, num_elements, spill, sc):
#         if self.key not in sc:
#             return
#  
#         if spill.volume is None:
#             raise ValueError("")
#  
#         _total_mass = spill.oil_props.get_density('kg/m^3') \
#             * self.get_volume('m^3') * 1000
#         sc[key][-num_elements:] = _total_mass / num_elements
#==============================================================================
