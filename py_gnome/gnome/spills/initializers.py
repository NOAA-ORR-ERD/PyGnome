#!/usr/bin/env python

"""
Utilities for initializing data arrays associated with the elements

This code is kept separately here to make it easier to mix and match
for various release and substance types.

"""

import numpy as np

from colander import SchemaNode, Int, Float, Range, TupleSchema

from gnome.utilities.rand import random_with_persistance
from gnome.array_types import gat

from gnome.cy_gnome.cy_rise_velocity_mover import rise_velocity_from_drop_size

from gnome.persist import base_schema
from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.utilities.distributions import (DistributionBase,
                                           NormalDistributionSchema,
                                           WeibullDistributionSchema,
                                           LogNormalDistributionSchema,
                                           UniformDistributionSchema)

class InitBaseClass(GnomeId):
    """
    All Init* classes will define the _state attribute, so just do so in a
    base class.

    It also documents that all initializers must implement an initialize
    method

    """

    def __init__(self, *args, **kwargs):
        # Contains the array_types that are set by an initializer but defined
        # anywhere else. For example, InitRiseVelFromDropletSizeFromDist()
        # computes droplet_diameter in the data_arrays even though it isn't
        # required by the mover. SpillContainer queries all initializers so it
        # knows about these array_types and can include them.
        # Make it a set since ElementType does a membership check in
        # set_newparticle_values()
        super(InitBaseClass, self).__init__(*args, **kwargs)

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        """
        all classes that derive from Base class must implement an initialize
        method.

        This method should initialize the appropriate data in
        the data arrays dict.

        See subclasses for examples.
        """
        pass


class WindageRangeSchema(TupleSchema):
    min_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.01)
    max_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.04)


class WindagesSchema(base_schema.ObjTypeSchema):
    """
    windages initializer values
    """
    windage_range = WindageRangeSchema(
        save=True, update=True,
    )
    windage_persist = SchemaNode(
        Int(), default=900, save=True, update=True,
    )


class InitWindages(InitBaseClass):
    _schema = WindagesSchema

    def __init__(self, windage_range=(0.01, 0.04), windage_persist=900, *args, **kwargs):
        """
        Initializes the windages, windage_range, windage_persist data arrays.
        Initial values for windages use infinite persistence. These are updated
        by the PointWindMover for particles with non-zero persistence.

        Optional arguments:

        :param windage_range: the windage range of the elements.
            Default is (0.01, 0.04) from 1% to 4%.
        :type windage_range: tuple: (min, max)

        :param windage_persist: Default is 900s, so windage is updated every
            900 seconds. -1 means the persistence is infinite so it is only set at
            the beginning of the run.
        :type windage_persist: integer seconds
        """
        super(InitWindages, self).__init__(*args, **kwargs)
        self.windage_persist = windage_persist
        self.windage_range = windage_range
        self.array_types.update({'windages': gat('windages'),
                                 'windage_range': gat('windage_range'),
                                 'windage_persist': gat('windage_persist')})

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'windage_range={0.windage_range}, '
                'windage_persist={0.windage_persist}'
                ')'.format(self))

    def to_dict(self, json_=None):
        return InitBaseClass.to_dict(self, json_=json_)

    def initialize(self, num_new_particles, data_arrays, substance):
        """
        Since windages exists in data_arrays, so must windage_range and
        windage_persist if this initializer is used/called
        """
        if any([k not in data_arrays for k in self.array_types.keys()]):
            return

        sl = slice(-num_new_particles, None, 1)
        data_arrays['windage_range'][sl] = self.windage_range
        data_arrays['windage_persist'][sl] = self.windage_persist
        random_with_persistance(
            data_arrays['windage_range'][-num_new_particles:, 0],
            data_arrays['windage_range'][-num_new_particles:, 1],
            data_arrays['windages'][-num_new_particles:]
        )


# do following two classes work for a time release spill?

#TODO: Get the distribution objects into this as first class objects, not
#shoehorned in the initialize()
class InitMassFromPlume(InitBaseClass):
    """
    Initialize the 'mass' array based on mass flux from the plume spilled
    """
    _schema = base_schema.ObjTypeSchema

    def __init__(self):
        """
        update array_types
        """
        super(InitMassFromPlume, self).__init__()
        self.array_types['mass'] = gat('mass')
        self.name = 'mass'

    def initialize(self, num_new_particles, data_arrays, substance):
        if any([k not in data_arrays for k in self.array_types.keys()]):
            return
        if substance.plume_gen is None:
            raise ValueError('plume_gen attribute of spill is None - cannot'
                             ' compute mass without plume mass flux')

        data_arrays['mass'][-num_new_particles:] = \
            substance.plume_gen.mass_of_an_le * 1000


class DistributionBaseSchema(base_schema.ObjTypeSchema):
    'Add schema to base class since all derived classes use same schema'

    # Fixme: IF we give all distributions he same API, this will be easier.
    distribution = GeneralGnomeObjectSchema(
        acceptable_schemas=[UniformDistributionSchema,
                            NormalDistributionSchema,
                            WeibullDistributionSchema,
                            LogNormalDistributionSchema],
        save=True, update=True
    )


class DistributionBase(InitBaseClass):
    '''
    Define a base class for all initializers that contain a distribution.
    Keep the code to serialize/deserialize distribution objects here so we only
    have to write it once.
    '''
    _schema = DistributionBaseSchema


class InitRiseVelFromDist(DistributionBase):

    def __init__(self, distribution=None, **kwargs):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        :param distribution: An initialized distribution object.
                             It should return values in m/s
        :type distribution: DistributionBase

        See gnome.utilities.distribution for details

        Right now, we have:

          * UniformDistribution
          * NormalDistribution
          * LogNormalDistribution
          * WeibullDistribution

        New distribution classes could be made. The only
        requirement is they need to have a set_values()
        method which accepts a NumPy array.
        (presumably, this function will also modify
        the array in some way)
        """
        super(InitRiseVelFromDist, self).__init__(**kwargs)

        if distribution and hasattr(distribution,"set_values"):
            self.distribution = distribution
        else:
            raise TypeError('InitRiseVelFromDist requires a distribution for '
                            'rise velocities')

        self.array_types['rise_vel'] = gat('rise_vel')
        self.name = 'rise_vel'

    def initialize(self, num_new_particles, data_arrays, substance):
        if any([k not in data_arrays for k in self.array_types.keys()]):
            return
        'Update values of "rise_vel" data array for new particles'
        self.distribution.set_values(
                            data_arrays['rise_vel'][-num_new_particles:])


class InitRiseVelFromDropletSizeFromDist(DistributionBase):
    #fixme: this does not seem to be tested.
    def __init__(self,
                 distribution=None,
                 water_density=1020.0, water_viscosity=1.0e-6,
                 **kwargs):
        """
        Set the droplet size from a distribution. Use the C++ get_rise_velocity
        function exposed via cython (rise_velocity_from_drop_size) to obtain
        rise_velocity from droplet size. Even though the droplet size is not
        changing over time, it is still stored in data array, as it can be
        useful for post-processing (called 'droplet_diameter')

        :param distribution: An object capable of generating a probability
                             distribution.
        :type distribution:

        Right now, we have:

         * UniformDistribution
         * NormalDistribution
         * LogNormalDistribution
         * WeibullDistribution

        New distribution classes could be made. The only
        requirement is they need to have a set_values()
        method which accepts a NumPy array.
        (presumably, this function will also modify
        the array in some way)

        :param water_density: 1020.0 [kg/m3]
        :type water_density: float
        :param water_viscosity: 1.0e-6 [m^2/s]
        :type water_viscosity: float
        """
        super(InitRiseVelFromDropletSizeFromDist, self).__init__(**kwargs)

        if distribution:
            self.distribution = distribution
        else:
            raise TypeError('InitRiseVelFromDropletSizeFromDist requires a '
                            'distribution for droplet sizes')

        self.water_viscosity = water_viscosity
        self.water_density = water_density
        self.array_types.update({'rise_vel': gat('rise_vel'),
                                 'droplet_diameter': gat('droplet_diameter')})
        self.name = 'rise_vel'

    def initialize(self, num_new_particles, data_arrays, substance):
        """
        Update values of 'rise_vel' and 'droplet_diameter' data arrays for
        new particles. First create a droplet_size array sampled from specified
        distribution, then use the cython wrapped (C++) function to set the
        'rise_vel' based on droplet size and properties like LE_density,
        water density and water_viscosity:
        gnome.cy_gnome.cy_rise_velocity_mover.rise_velocity_from_drop_size()
        """
        if any([k not in data_arrays for k in self.array_types.keys()]):
            return
        drop_size = np.zeros((num_new_particles, ), dtype=np.float64)
        le_density = np.zeros((num_new_particles, ), dtype=np.float64)

        self.distribution.set_values(drop_size)

        data_arrays['droplet_diameter'][-num_new_particles:] = drop_size

        # Don't require a water object
        # water_temp = spill.water.get('temperature')
        # le_density[:] = substance.density_at_temp(water_temp)
        if hasattr(substance, 'water'):
            water = substance.water
        else:
            water = None
        if water is not None:
            water_temp = water.get('temperature')
            le_density[:] = substance.density_at_temp(water_temp)
        else:
            le_density[:] = substance.density_at_temp()

        # now update rise_vel with droplet size - dummy for now
        rise_velocity_from_drop_size(
                                data_arrays['rise_vel'][-num_new_particles:],
                                le_density, drop_size,
                                self.water_viscosity, self.water_density)


# def floating_initializers(windage_range=(.01, .04),
#                           windage_persist=900,):
#     """
#     Helper function returns a list of initializers for floating LEs

#     1. InitWindages(): for initializing 'windages' with user specified
#     windage_range and windage_persist.

#     :param substance='oil_conservative': Type of oil spilled. Passed onto
#         ElementType constructor
#     :type substance: str or OilProps

#     fixme: maybe just have this be a windage initializer??
#     """
#     return [InitWindages(windage_range=windage_range,
#                          windage_persist=windage_persist)]


def plume_initializers(distribution_type='droplet_size',
                       distribution=None,
                       windage_range=(.01, .04),
                       windage_persist=900,
                       **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages' initialized with user specified parameters for distribution.

    :param str distribution_type: type of distribution, 'droplet_size' or 'rise_velocity'

    :param gnome.utilities.distributions distribution=None:

    :param windage_range: minimum and maximum windage
    :type windage_range: tuple-of-floats

    :param int windage_persist: persistence of windage in seconds

    :param str substance_name=None:

    :param float density=None:

    :param str density_units='kg/m^3':

    Distribution type available options:

     * 'droplet_size': Droplet size is sampled from the specified distribution.
                       No droplet size is computed.
     * 'rise_velocity': Rise velocity is directly sampled from the specified
                        distribution.  Rise velocity is calculated.

    Distributions - An object capable of generating a probability distribution.

    Right now, we have:

     * UniformDistribution
     * NormalDistribution
     * LogNormalDistribution
     * WeibullDistribution

    New distribution classes could be made. The only requirement is they
    need to have a ``set_values()`` method which accepts a NumPy array.
    (presumably, this function will also modify the array in some way)

    .. note:: substance_name or density must be provided

    """
    # Add docstring from called classes
    # Note: following gives sphinx warnings on build, ignore for now.

    plume_initializers.__doc__ += ("\nInitRiseVelFromDropletSizeFromDist Documentation:\n" +
                      InitRiseVelFromDropletSizeFromDist.__init__.__doc__ +
                      "\nInitRiseVelFromDist Documentation:\n" +
                      InitRiseVelFromDist.__init__.__doc__ +
                      "\nInitWindages Documentation:\n" +
                      InitWindages.__init__.__doc__
                      )

    if distribution_type == 'droplet_size':
        return [InitRiseVelFromDropletSizeFromDist(distribution=distribution, **kwargs),
                InitWindages(windage_range, windage_persist)]
    elif distribution_type == 'rise_velocity':
        return [InitRiseVelFromDist(distribution=distribution,**kwargs),
                InitWindages(windage_range, windage_persist)]
    else:
        raise TypeError('distribution_type must be either droplet_size or '
                        'rise_velocity')


def plume_from_model_initializers(distribution_type='droplet_size',
                                  distribution=None,
                                  windage_range=(.01, .04),
                                  windage_persist=900,
                                  **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages'
    initializer with user specified parameters for distribution.
    """
    if distribution_type == 'droplet_size':
        return [InitRiseVelFromDropletSizeFromDist(distribution=distribution, **kwargs),
                InitWindages(windage_range, windage_persist),
                InitMassFromPlume()]
    elif distribution_type == 'rise_velocity':
        return [InitRiseVelFromDist(distribution=distribution, **kwargs),
                InitWindages(windage_range, windage_persist),
                InitMassFromPlume()]
