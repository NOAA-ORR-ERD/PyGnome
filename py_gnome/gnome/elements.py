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
        """
        Initializes the windages, windage_range, windage_persist data arrays.
        Initial values for windages use infinite persistence. These are updated
        by the WindMover for particles with non-zero persistence.

        Optional arguments:

        :param windage_range=(0.01, 0.04): the windage range of the elements
            default is (0.01, 0.04) from 1% to 4%.
        :type windage_range: tuple: (min, max)

        :param windage_persist=-1: Default is 900s, so windage is updated every
            900 sec. -1 means the persistence is infinite so it is only set at
            the beginning of the run.
        :type windage_persist: integer seconds
        """
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
            raise ValueError("'windage_range' >= (0, 0). Nominal values vary"
                " between 1% to 4%, so default windage_range=(0.01, 0.04)")
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
    Initialize the 'mass' array based on total volume spilled and the type of
    substance
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
    def __init__(self, **kwargs):
        """
        Values to be sampled from a distribution.

        Keyword arguments: kwargs are different based on the type of
        distribution selected by user

        :param distribution: could be 'uniform', 'normal', 'lognormal' or 'weibull'
        :type distribution: str

        If distribution is 'uniform', then following kwargs are expected

        :param low: for 'uniform' dist, it is lower bound. Default is 0.
        :param high: for 'uniform' dist, it is upper bound. Default is 0.1

        If distribution is 'normal' or 'lognormal', then following kwargs are
        expected

        :param mean: mean of the normal distribution
        :param sigma: 1 standard deviation (sigma) of normal distribution

        If distribution is 'weibull', then following kwargs are expected.

        :param alpha: shape parameter 'alpha' - labeled as 'a' in
            numpy.random.weibull distribution
        :param lambda_: the scale parameter for the distribution - required for
            2-parameter weibull distribution (Rosin-Rammler). Default is 1.

        """
        distribution = kwargs.pop('distribution', 'uniform')
        if distribution not in ['uniform', 'normal', 'lognormal', 'weibull']:
            raise ValueError("{0} is unknown distribution. Only 'uniform',"
                             " 'normal', 'lognormal', and 'weibull'"
                             " distributions are implemented")

        self.distribution = distribution

        if distribution == 'uniform':
            self.low = kwargs.pop('low', 0)
            self.high = kwargs.pop('high', .1)
        elif distribution in ['normal', 'lognormal']:
            self.mean = kwargs.pop('mean', None)
            self.sigma = kwargs.pop('sigma', None)
            if self.mean is None or self.sigma is None:
                raise TypeError("'normal' distribution requires 'mean' and"
                                " 'sigma' input as kwargs")
        elif distribution == 'weibull':
            self.alpha = kwargs.pop('alpha', None)
            self.lambda_ = kwargs.pop('lambda_', 1)
            if self.alpha is None:
                raise TypeError("'weibull' distribution requires 'alpha'"
                                " input as kwargs")

    def set_values(self, np_array):
        """
        Takes a numpy array as input and fills it with values generated from
        specified distribution

        :param np_array: numpy array to be filled with values sampled from
            specified distribution
        :type np_array: numpy array of dtype 'float64'
        """
        if self.distribution == 'uniform':
            np_array[:] = np.random.uniform(self.low, self.high,
                                         len(np_array))
        elif self.distribution == 'normal':
            np_array[:] = np.random.normal(self.mean, self.sigma,
                                        len(np_array))
        elif self.distribution == 'lognormal':
            np_array[:] = np.random.lognormal(self.mean, self.sigma,
                                        len(np_array))
        elif self.distribution == 'weibull':
            np_array[:] = (self.lambda_ *
                           np.random.weibull(self.alpha, len(np_array)))


class InitRiseVelFromDist(ValuesFromDistBase):
    def __init__(self, distribution='uniform', **kwargs):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        Use distribution to define rise_vel - use super to invoke
        ValuesFromDistBase().__init__()

        :param distribution: could be 'uniform', 'normal', 'lognormal' or 'weibull'
        :type distribution: str

        If distribution is 'uniform', then following kwargs are expected

        :param low: for 'uniform' dist, it is lower bound. Default is 0.
        :param high: for 'uniform' dist, it is upper bound. Default is 0.1

        If distribution is 'normal' or 'lognormal', then following kwargs are
        expected

        :param mean: mean of the normal distribution
        :param sigma: 1 standard deviation (sigma) of normal distribution

        If distribution is 'weibull', then following kwargs are expected.

        :param alpha: shape parameter 'alpha' - labeled as 'a' in
            numpy.random.weibull distribution
        :param lambda_: the scale parameter for the distribution - required for
            2-parameter weibull distribution (Rosin-Rammler). Default is 1.
        """

        super(InitRiseVelFromDist, self).__init__(distribution=distribution,
                                                  **kwargs)

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
        """
        Update values of 'rise_vel' data array for new particles
        """
        self.set_values(data_arrays['rise_vel'][-num_new_particles:])


class InitRiseVelFromDropletSizeFromDist(ValuesFromDistBase):
    def __init__(self,
                 distribution='uniform',
                 water_density=1020.0,
                 water_viscosity=1.0e-6,
                 **kwargs):
        """
        Set the droplet size from a distribution. Use the C++ get_rise_velocity
        function exposed via cython (rise_velocity_from_drop_size) to obtain
        rise_velocity from droplet size. In this case, the droplet size is not
        changing over time, so no data array for droplet size exists.

        Use distribution to define rise_vel - use super to invoke
        ValuesFromDistBase().__init__()

        All parameters have defaults and are optional

        :param distribution: could be 'uniform', 'normal' ,'lognormal' or 'weibull'.
            Default value 'uniform'
        :type distribution: str

        If distribution is 'uniform', then following kwargs are expected

        :param low: for 'uniform' dist, it is lower bound. Default is 0.
        :param high: for 'uniform' dist, it is upper bound. Default is 0.1

        If distribution is 'normal' or 'lognormal', then following kwargs are
        expected

        :param mean: mean of the normal distribution
        :param sigma: 1 standard deviation (sigma) of normal distribution

        If distribution is 'weibull', then following kwargs are expected.

        :param alpha: shape parameter 'alpha' - labeled as 'a' in
            numpy.random.weibull distribution
        :param lambda_: the scale parameter for the distribution - required for
            2-parameter weibull distribution (Rosin-Rammler). Default is 1.

        :param water_density: 1020.0 [kg/m3]
        :type water_density: float
        :param water_viscosity: 1.0e-6 [m^2/s]
        :type water_viscosity: float
        """
        super(InitRiseVelFromDropletSizeFromDist, self).__init__(
                                        distribution=distribution, **kwargs)
        self.water_viscosity = water_viscosity
        self.water_density = water_density

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        """
        Update values of 'rise_vel' data array for new particles. First
        create a droplet_size array sampled from specified distribution, then
        use the cython wrapped (C++) function to set the 'rise_vel' based on
        droplet size and properties like LE_density, water density and
        water_viscosity:
        gnome.cy_gnome.cy_rise_velocity_mover.rise_velocity_from_drop_size()
        """
        drop_size = np.zeros((num_new_particles, ), dtype=np.float64)
        le_density = np.zeros((num_new_particles, ), dtype=np.float64)

        self.set_values(drop_size)
        le_density[:] = substance.density

        # now update rise_vel with droplet size - dummy for now
        rise_velocity_from_drop_size(
                                data_arrays['rise_vel'][-num_new_particles:],
                                le_density,
                                drop_size,
                                self.water_viscosity,
                                self.water_density)


""" ElementType classes"""


class ElementType(object):
    def __init__(self, initializers, substance='oil_conservative'):
        """
        Define initializers for the type of elements

        :param initializers:
        :type initializers: dict

        :param substance='oil_conservative': Type of oil spilled.
            If this is a string, or an oillibrary.models.Oil object, then
            create gnome.spill.OilProps(oil) object. If this is a
            gnome.spill.OilProps object, then simply instance oil_props
            variable to it: self.oil_props = oil
        :type substance: either str, or oillibrary.models.Oil object or
            gnome.spill.OilProps
        """
        self.initializers = initializers
        if isinstance(substance, basestring):
            self.substance = OilProps(substance)
        else:
            # assume object passed in is duck typed to be same as OilProps
            self.substance = substance

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        """
        call all initializers. This will set the initial values for all
        data_arrays.
        """
        if num_new_particles > 0:
            for key in data_arrays:
                if key in self.initializers:
                    self.initializers[key].initialize(num_new_particles,
                                                      spill,
                                                      data_arrays,
                                                      self.substance)


def floating(windage_range=(.01, .04), windage_persist=900):
    """
    Helper function returns an ElementType object containing 'windages'
    initializer with user specified windage_range and windage_persist.
    """
    return ElementType({'windages': InitWindages(windage_range,
                                                 windage_persist)})

def plume(distribution_type='droplet_size', distribution='weibull', windage_range=(.01, .04), windage_persist=900, **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel' and 'windages'
    initializer with user specified parameters for distribution.
    """ 
    if distribution_type == 'droplet_size':
        return ElementType({'rise_vel': InitRiseVelFromDropletSizeFromDist(distribution=distribution,
                                                 **kwargs),
                                                 'windages': InitWindages(windage_range,windage_persist),
                                                 'mass': InitMassFromVolume()})
    elif distribution_type == 'rise_velocity':
        return ElementType({'rise_vel': InitRiseVelFromDist(distribution=distribution,
                                                 **kwargs),
                                                 'windages': InitWindages(windage_range,windage_persist),
                                                 'mass': InitMassFromVolume(windage_range,windage_persist)})
