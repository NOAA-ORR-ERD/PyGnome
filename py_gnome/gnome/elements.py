'''
Types of elements that a spill can expect
These are properties that are spill specific like:
  'floating' element_types would contain windage_range, windage_persist
  'subsurface_dist' element_types would contain rise velocity distribution info
  'nonweathering' element_types would set use_droplet_size flag to False
  'weathering' element_types would use droplet_size, densities, mass?
'''
import numpy as np


"""
Initializers for various element types
Each initializer only sets the value for a single numpy array
"""


class InitConstantWindageRange(object):
    """
    initialize floating elements with windage range information
    """
    def __init__(self, windage_range=[0.01, 0.04]):
        self.windage_range = windage_range

    def initialize(self, num_new_particles, spill, data_arrays):
        """
        Since windages exists in data_arrays, so must windage_range and
        windage_persist if this initializer is used/called
        """
        data_arrays['windage_range'][-num_new_particles:, :] = \
            self.windage_range


class InitConstantWindagePersist(object):
    """
    initialize floating elements with windage persistance information
    """
    def __init__(self, windage_persist=900):
        self.windage_persist = windage_persist

    def initialize(self, num_new_particles, spill, data_arrays):
        """
        Since windages exists in data_arrays, so must windage_range and
        windage_persist if this initializer is used/called
        """
        data_arrays['windage_persist'][-num_new_particles:] = \
            self.windage_persist


class InitMassFromVolume(object):
    """
    This could also go into ArrayType's initialize method if we pass
    spill as input to initialize method
    """
    def initialize(self, num_new_particles, spill, data_arrays):
        if spill.volume is None:
            raise ValueError('volume attribute of spill is None - cannot'
                             ' compute mass without volume')

        _total_mass = spill.oil_props.get_density('kg/m^3') \
            * spill.get_volume('m^3') * 1000
        data_arrays['mass'][-num_new_particles:] = (_total_mass /
                                                    num_new_particles)


class InitRiseVelFromDist(object):
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

    def initialize(self, num_new_particles, spill, data_arrays):
        """
        if 'rise_vel' exists in SpillContainer's data_arrays, then define
        """
        if self.distribution == 'uniform':
            data_arrays['rise_vel'][-num_new_particles:] = np.random.uniform(
                                                        self.params[0],
                                                        self.params[1],
                                                        num_new_particles)
        elif self.distribution == 'normal':
            data_arrays['rise_vel'][-num_new_particles:] = np.random.normal(
                                                        self.params[0],
                                                        self.params[1],
                                                        num_new_particles)

""" ElementType classes"""


class ElementType(object):
    def __init__(self):
        self.initializers = {}

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        if num_new_particles > 0:
            for key in data_arrays:
                if key in self.initializers:
                    self.initializers[key].initialize(num_new_particles, spill,
                                                      data_arrays)


class Floating(ElementType):
    def __init__(self):
        """
        Mover should define windages, windage_range and windage_persist
        """
        self.initializers = {'windage_range': InitConstantWindageRange(),
                             'windage_persist': InitConstantWindagePersist()}


class FloatingMassFromVolume():
    def __init__(self):
        self.initializers = {'windage_range': InitConstantWindageRange(),
                             'windage_persist': InitConstantWindagePersist(),
                             'mass': InitMassFromVolume()}


class FloatingMassFromVolumeRiseVel():
    def __init__(self):
        self.initializers = {'windage_range': InitConstantWindageRange(),
                             'windage_persist': InitConstantWindagePersist(),
                             'mass': InitMassFromVolume(),
                             'rise_vel': InitRiseVelFromDist()}