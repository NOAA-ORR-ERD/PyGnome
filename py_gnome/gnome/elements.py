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
from gnome.cy_gnome.cy_rise_velocity_mover import rise_velocity_from_drop_size
from gnome.db.oil_library.oil_props import OilProps
"""
Initializers for various element types
"""


class InitWindages(object):
    def __init__(self, windage_range=(0.01, 0.04), windage_persist=900):
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

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
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

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        if spill.volume is None:
            raise ValueError('volume attribute of spill is None - cannot'
                             ' compute mass without volume')

        _total_mass = substance.get_density('kg/m^3') \
            * spill.get_volume('m^3') * 1000
        data_arrays['mass'][-num_new_particles:] = (_total_mass /
                                                    num_new_particles)


class ValuesFromDistBase(object):
    def __init__(self, distribution='uniform', params=(0, .1)):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        Use distribution to define rise_vel - use super to invoke
        ValuesFromDistBase().__init__()

        :param distribution: could be 'uniform' or 'normal'
        :type distribution: str

        :param params: for 'uniform' dist, it is (min_val, max_val).
            For 'normal' dist, it is (mean, sigma) where sigma is
            1 standard deviation
        :type params: list of length 2
        """

        if distribution not in ['uniform', 'normal']:
            raise ValueError("{0} is unknown distribution. Only 'uniform' or"
                             " 'normal' distribution is implemented")

        self.distribution = distribution
        self.params = params

    def set_values(self, np_array):
        if self.distribution == 'uniform':
            np_array[:] = np.random.uniform(self.params[0], self.params[1],
                                         len(np_array))
        elif self.distribution == 'normal':
            np_array[:] = np.random.normal(self.params[0], self.params[1],
                                        len(np_array))


class InitRiseVelFromDist(ValuesFromDistBase):
    def __init__(self, distribution='uniform', params=(0, .1)):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        Use distribution to define rise_vel - use super to invoke
        ValuesFromDistBase().__init__()

        :param distribution: could be 'uniform' or 'normal'
        :type distribution: str

        :param params: for 'uniform' dist, it is (min_val, max_val).
            For 'normal' dist, it is (mean, sigma) where sigma is
            1 standard deviation
        :type params: list of length 2
        """

        super(InitRiseVelFromDist, self).__init__(distribution, params)

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
        """
        if 'rise_vel' exists in SpillContainer's data_arrays, then define
        """
        self.set_values(data_arrays['rise_vel'][-num_new_particles:])


class InitRiseVelFromDropletSizeFromDist(ValuesFromDistBase):
    def __init__(self,
                 distribution='uniform',
                 params=(0, .1),
                 water_density=1020.0,
                 water_viscosity=1.0e-6):
        """
        Set the droplet size from a distribution. Use the C++ get_rise_velocity
        function exposed via cython (rise_velocity_from_drop_size) to obtain
        rise_velocity from droplet size. In this case, the droplet size is not
        changing over time, so no data array for droplet size exists.

        Use distribution to define rise_vel - use super to invoke
        ValuesFromDistBase().__init__()

        All parameters have defaults and are optional

        :param distribution: could be 'uniform' or 'normal'.
            Default value 'uniform'
        :type distribution: str

        :param params: for 'uniform' dist, it is (min_val, max_val).
            For 'normal' dist, it is (mean, sigma) where sigma is
            1 standard deviation. Default value (0, .1)
        :type params: list of length 2

        :param water_density: 1020.0 [kg/m3]
        :type water_density: float
        :param water_viscosity: 1.0e-6 [m^2/s]
        :type water_viscosity: float
        """
        super(InitRiseVelFromDropletSizeFromDist, self).__init__(distribution,
                                                                 params)
        self.water_viscosity = water_viscosity
        self.water_density = water_density

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
        """
        if 'rise_vel' exists in SpillContainer's data_arrays, then define
        """
        drop_size = np.zeros((num_new_particles, ), dtype=np.float64)
        le_density = np.zeros((num_new_particles, ), dtype=np.float64)

        self.set_values(drop_size)
        le_density[:] = spill.oil_props.density

        # now update rise_vel with droplet size - dummy for now
        rise_velocity_from_drop_size(
                                data_arrays['rise_vel'][-num_new_particles:],
                                le_density,
                                drop_size,
                                self.water_viscosity,
                                self.water_density)


""" ElementType classes"""


class ElementType(object):
    def __init__(self, initializers={}, substance='oil_conservative'):
        self.initializers = initializers
        if isinstance(substance, basestring):
            self.substance = OilProps(substance)
        else:
            # assume object passed in is duck typed to be same as OilProps
            self.substance = substance

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        if num_new_particles > 0:
            for key in data_arrays:
                if key in self.initializers:
                    self.initializers[key].initialize(num_new_particles,
                                                      spill,
                                                      data_arrays,
                                                      self.substance)


def floating(windage_range=(.01, .04), windage_persist=900):
    return ElementType({'windages': InitWindages(windage_range,
                                                            windage_persist)})
