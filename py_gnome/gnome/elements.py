'''
Types of elements that a spill can expect
These are properties that are spill specific like:
  'floating' element_types would contain windage_range, windage_persist
  'subsurface_dist' element_types would contain rise velocity distribution info
  'nonweathering' element_types would set use_droplet_size flag to False
  'weathering' element_types would use droplet_size, densities, mass?
'''
import numpy as np
from gnome.utilities.rand import random_with_persistance

"""
Initializers for various element types
"""


class InitWindagesConstantParams(object):
    def __init__(self, windage_range=[0.01, 0.04], windage_persist=900):
        self.windage_persist = windage_persist
        self.windage_range = windage_range

    @property
    def windage_persist(self):
        return self._windage_persist

    @windage_persist.setter
    def windage_persist(self, val):
        if val == 0:
            raise ValueError("'windage_persist' cannot be 0. For infinite"
                " windage, windage_persist=-1 otherwise windage_persist > 0.")
        self._windage_persist = val

    @property
    def windage_range(self):
        return self._windage_range

    @windage_range.setter
    def windage_range(self, val):
        if np.any(np.asarray(val) < 0):
            raise ValueError("'windage_range' > [0, 0]. Nominal values vary"
                " between 1% to 4%, so default windage_range = [0.01, 0.04]")
        self._windage_range = val

    def initialize(self, num_new_particles, spill, data_arrays):
        """
        Since windages exists in data_arrays, so must windage_range and
        windage_persist if this initializer is used/called
        """
        data_arrays['windage_range'][-num_new_particles:, 0] = \
            self.windage_range[0]
        data_arrays['windage_range'][-num_new_particles:, 1] = \
            self.windage_range[1]
        data_arrays['windage_persist'][-num_new_particles:] = \
            self.windage_persist

        # initialize all windages - ignore persistence during initialization
        # if we have infinite persistence, these values are never updated
        random_with_persistance(
                    data_arrays['windage_range'][-num_new_particles:][:, 0],
                    data_arrays['windage_range'][-num_new_particles:][:, 1],
                    data_arrays['windages'][-num_new_particles:])


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
    def __init__(self, distribution='uniform', params=[0, .1]):
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
        For this ElementType, all three must be defined!
        """
        super(Floating, self).__init__()
        self.initializers = {'windages': InitWindagesConstantParams()}


class FloatingMassFromVolume(Floating):
    def __init__(self):
        super(FloatingMassFromVolume, self).__init__()
        self.initializers.update({'mass': InitMassFromVolume()})


class FloatingWithRiseVel(Floating):
    def __init__(self):
        super(FloatingWithRiseVel, self).__init__()
        self.initializers.update({'rise_vel': InitRiseVelFromDist()})


class FloatingMassFromVolumeRiseVel(FloatingMassFromVolume):
    def __init__(self):
        super(FloatingMassFromVolumeRiseVel, self).__init__()
        self.initializers.update({'rise_vel': InitRiseVelFromDist()})
