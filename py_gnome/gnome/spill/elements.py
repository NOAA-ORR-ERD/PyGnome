#!/usr/bin/env python
'''
Types of elements that a spill can expect
These are properties that are spill specific like:
  'floating' element_types would contain windage_range, windage_persist
  'subsurface_dist' element_types would contain rise velocity distribution info
  'nonweathering' element_types would set use_droplet_size flag to False
  'weathering' element_types would use droplet_size, densities, mass?
'''
import copy
import numpy
np = numpy
from colander import SchemaNode, Int, Float, Range, TupleSchema

import gnome    # required by new_from_dict
from gnome.utilities.rand import random_with_persistance
from gnome.utilities.compute_fraction import fraction_below_d
from gnome.utilities.serializable import Serializable
from gnome.utilities.distributions import UniformDistribution

from gnome.cy_gnome.cy_rise_velocity_mover import rise_velocity_from_drop_size
from gnome.db.oil_library.oil_props import (OilProps, OilPropsFromDensity)

from gnome.persist import base_schema
"""
Initializers for various element types
"""


class InitBaseClass(object):
    """
    All Init* classes will define the _state attribute, so just do so in a
    base class.

    It also documents that all initializers must implement an initialize method

    todo/Note:
    This may change as the persistence code changes. Currently, 'id' and
    'obj_type' are part of base Serializable._state
    """
    _state = copy.deepcopy(Serializable._state)

    def initialize(self):
        """
        all classes that derive from Base class must implement initialize
        method
        """
        pass


class WindageSchema(TupleSchema):
    min_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.01)
    max_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.04)
    name = 'windage_range'


class InitWindagesSchema(base_schema.ObjType):
    """
    windages initializer values
    """
    windage_range = WindageSchema()
    windage_persist = SchemaNode(Int(), default=900,
                                 description='windage persistence in minutes')
    name = 'windages'


class InitWindages(InitBaseClass, Serializable):
    _update = ['windage_range', 'windage_persist']
    _create = []
    _create.extend(_update)
    _state = copy.deepcopy(InitBaseClass._state)
    _state.add(save=_create, update=_update)
    _schema = InitWindagesSchema

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

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'windage_range={0.windage_range}, '
                'windage_persist={0.windage_persist}'
                ')'.format(self))

    @property
    def windage_persist(self):
        return self._windage_persist

    @windage_persist.setter
    def windage_persist(self, val):
        if val == 0:
            raise ValueError("'windage_persist' cannot be 0. "
                             "For infinite windage, windage_persist=-1 "
                             "otherwise windage_persist > 0.")
        self._windage_persist = val

    @property
    def windage_range(self):
        return self._windage_range

    @windage_range.setter
    def windage_range(self, val):
        if np.any(np.asarray(val) < 0) or np.asarray(val).size != 2:
            raise ValueError("'windage_range' >= (0, 0). "
                             "Nominal values vary between 1% to 4%. "
                             "Default windage_range=(0.01, 0.04)")
        self._windage_range = val

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
        """
        Since windages exists in data_arrays, so must windage_range and
        windage_persist if this initializer is used/called
        """
        (data_arrays['windage_range'][-num_new_particles:, 0],
         data_arrays['windage_range'][-num_new_particles:, 1],
         data_arrays['windage_persist'][-num_new_particles:]) = \
            (self.windage_range[0],
             self.windage_range[1],
             self.windage_persist)

        # initialize all windages - ignore persistence during initialization
        # if we have infinite persistence, these values are never updated
        random_with_persistance(
                    data_arrays['windage_range'][-num_new_particles:][:, 0],
                    data_arrays['windage_range'][-num_new_particles:][:, 1],
                    data_arrays['windages'][-num_new_particles:])


class InitMassComponentsFromOilProps(InitBaseClass, Serializable):
    '''
       Initialize the mass components based on given Oil properties
    '''
    _state = copy.deepcopy(InitBaseClass._state)
    _schema = base_schema.ObjType

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        '''
           :param int num_new_particles: Number of new particles to initialize
           :param Spill spill: The spill object from which the new particles
                               are coming from.
           :param data_arrays: The numpy arrays that make up the collective
                               properties of our particles.
           :type data_arrays: dict(<name>: <np.ndarray>,
                                   ...
                                   )
           :param OilProps substance: The Oil Properties associated with the
                                      spill.
                                      (TODO: Why is this not simply contained
                                             in the Spill??
                                             Why the extra argument??)
        '''
        if spill.mass is None:
            raise ValueError('mass attribute of spill is None - cannot '
                             'compute particle mass without total mass')

        total_mass = spill.get_mass('g')
        le_mass = total_mass / spill.release.num_elements

        mass_fractions = np.asarray(zip(*substance.mass_components)[0],
                                    dtype=np.float64)
        masses = mass_fractions * le_mass

        data_arrays['mass_components'][-num_new_particles:] = masses


class InitHalfLivesFromOilProps(InitBaseClass, Serializable):
    '''
       Initialize the half-lives of our mass components based on given Oil
       properties.
    '''
    _state = copy.deepcopy(InitBaseClass._state)
    _schema = base_schema.ObjType

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        '''
           :param int num_new_particles: Number of new particles to initialize
           :param Spill spill: The spill object from which the new particles
                               are coming from.
           :param data_arrays: The numpy arrays that make up the collective
                               properties of our particles.
           :type data_arrays: dict(<name>: <np.ndarray>,
                                   ...
                                   )
           :param OilProps substance: The Oil Properties associated with the
                                      spill.
                                      (TODO: Why is this not simply contained
                                             in the Spill??
                                             Why the extra argument??)
        '''
        half_lives = np.asarray(zip(*substance.mass_components)[1],
                                dtype=np.float64)

        data_arrays['half_lives'][-num_new_particles:] = half_lives


# do following two classes work for a time release spill?


class InitMassFromTotalMass(InitBaseClass, Serializable):
    """
    Initialize the 'mass' array based on total mass spilled.
    todo: are both InitMassFromTotalMass and InitMassFromTotalVolume required?
    If one is known, isn't the other one known too?
    """

    _state = copy.deepcopy(InitBaseClass._state)
    _schema = base_schema.ObjType

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        if spill.mass is None:
            raise ValueError('mass attribute of spill is None - cannot '
                             'compute particle mass without total mass')

        _total_mass = spill.get_mass('g')
        data_arrays['mass'][-num_new_particles:] = (_total_mass /
                                                    spill.release.num_elements)


# NOT REQUIRED. IF DENSITY IS KNOWN, WE CAN COMPUTE TOTAL MASS IN SPILL - THEN
# USE InitMassFromTotalMass as initializer
#==============================================================================
# class InitMassFromVolume(InitBaseClass, Serializable):
#     """
#     Initialize the 'mass' array based on total volume spilled and the type of
#     substance. No parameters, as it uses the volume specified elsewhere.
#     """
#     _state = copy.deepcopy(InitBaseClass._state)
#     _schema = base_schema.ObjType
# 
#     def initialize(self, num_new_particles, spill, data_arrays, substance):
#         if spill.volume is None:
#             raise ValueError('volume attribute of spill is None - cannot '
#                              'compute mass without volume')
# 
#         _total_mass = (substance.get_density('kg/m^3')
#                        * spill.get_volume('m^3') * 1000)
#         data_arrays['mass'][-num_new_particles:] = (_total_mass /
#                                                     spill.release.num_elements)
#==============================================================================


class InitMassFromPlume(InitBaseClass, Serializable):
    """
    Initialize the 'mass' array based on mass flux from the plume spilled
    """
    _state = copy.deepcopy(InitBaseClass._state)
    _schema = base_schema.ObjType

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        if spill.plume_gen is None:
            raise ValueError('plume_gen attribute of spill is None - cannot'
                             ' compute mass without plume mass flux')

        data_arrays['mass'][-num_new_particles:] = \
            spill.plume_gen.mass_of_an_le * 1000


class DistributionBaseSchema(base_schema.ObjType):
    'Add schema to base class since all derived classes use same schema'
    description = 'dynamically adds distribution schema to self'

    def __init__(self, **kwargs):
        dist = kwargs.pop('distribution')
        self.add(dist)
        super(DistributionBaseSchema, self).__init__(**kwargs)


class DistributionBase(InitBaseClass, Serializable):
    '''
    Define a base class for all initializers that contain a distribution.
    Keep the code to serialize/deserialize distribution objects here so we only
    have to write it once.
    '''
    _state = copy.deepcopy(InitBaseClass._state)
    _state.add(save=['distribution'], update=['distribution'])
    _schema = DistributionBaseSchema

    def serialize(self, json_='webapi'):
        'Add distribution schema based on "distribution" - then serialize'

        dict_ = self.to_serialize(json_)
        schema = self.__class__._schema(name=self.__class__.__name__,
                   distribution=self.distribution._schema(name='distribution'))
        return schema.serialize(dict_)

    @classmethod
    def deserialize(cls, json_):
        'Add distribution schema based on "distribution" - then deserialize'
        dist_type = json_['distribution']['obj_type']
        to_eval = "{0}._schema(name='distribution')".format(dist_type)
        dist_schema = eval(to_eval)
        schema = cls._schema(name=cls.__name__, distribution=dist_schema)
        dict_ = schema.deserialize(json_)

        # convert nested object ['distribution'] saved as a
        # dict, back to an object if json_ == 'save' here itself
        if json_['json_'] == 'save':
            distribution = dict_.get('distribution')
            to_eval = '{0}.new_from_dict(distribution)'.format(
                                                    distribution['obj_type'])
            dict_['distribution'] = eval(to_eval)

        return dict_


class InitRiseVelFromDist(DistributionBase):
    _state = copy.deepcopy(DistributionBase._state)

    def __init__(self, distribution=None, **kwargs):
        """
        Set the rise velocity parameters to be sampled from a distribution.

        Use distribution to define rise_vel

        :param distribution: An object capable of generating a probability
                             distribution.
        :type distribution: Right now, we have:
                              - UniformDistribution
                              - NormalDistribution
                              - LogNormalDistribution
                              - WeibullDistribution
                            New distribution classes could be made.  The only
                            requirement is they need to have a set_values()
                            method which accepts a NumPy array.
                            (presumably, this function will also modify
                             the array in some way)
        """
        super(InitRiseVelFromDist, self).__init__(**kwargs)

        if distribution:
            self.distribution = distribution
        else:
            self.distribution = UniformDistribution()

    def initialize(self, num_new_particles, spill, data_arrays,
                   substance=None):
        'Update values of "rise_vel" data array for new particles'
        self.distribution.set_values(
                            data_arrays['rise_vel'][-num_new_particles:])


class InitRiseVelFromDropletSizeFromDist(DistributionBase):
    _state = copy.deepcopy(DistributionBase._state)

    def __init__(self, distribution=None,
                 water_density=1020.0, water_viscosity=1.0e-6,
                 **kwargs):
        """
        Set the droplet size from a distribution. Use the C++ get_rise_velocity
        function exposed via cython (rise_velocity_from_drop_size) to obtain
        rise_velocity from droplet size. Even though the droplet size is not
        changing over time, it is still stored in data array, as it can be
        useful for post-processing (called 'droplet_diameter')

        Use distribution to define rise_vel

        :param distribution: An object capable of generating a probability
                             distribution.
        :type distribution: Right now, we have:
                              - UniformDistribution
                              - NormalDistribution
                              - LogNormalDistribution
                              - WeibullDistribution
                            New distribution classes could be made.  The only
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
            self.distribution = UniformDistribution()

        self.water_viscosity = water_viscosity
        self.water_density = water_density

    def initialize(self, num_new_particles, spill, data_arrays, substance):
        """
        Update values of 'rise_vel' and 'droplet_diameter' data arrays for
        new particles. First create a droplet_size array sampled from specified
        distribution, then use the cython wrapped (C++) function to set the
        'rise_vel' based on droplet size and properties like LE_density,
        water density and water_viscosity:
        gnome.cy_gnome.cy_rise_velocity_mover.rise_velocity_from_drop_size()
        """
        drop_size = np.zeros((num_new_particles, ), dtype=np.float64)
        le_density = np.zeros((num_new_particles, ), dtype=np.float64)

        self.distribution.set_values(drop_size)

        data_arrays['droplet_diameter'][-num_new_particles:] = drop_size
        le_density[:] = substance.density

        # now update rise_vel with droplet size - dummy for now
        rise_velocity_from_drop_size(
                                data_arrays['rise_vel'][-num_new_particles:],
                                le_density, drop_size,
                                self.water_viscosity, self.water_density)


""" ElementType classes"""


class ElementType(Serializable):
    _state = copy.deepcopy(Serializable._state)
    _state.add(save=['initializers'], update=['initializers'])
    _schema = base_schema.ObjType

    def __init__(self, initializers, substance='oil_conservative'):
        '''
        Define initializers for the type of elements

        :param dict initializers: a dict of initializers where the keys
            correspond with names in data_arrays (stored in SpillContainer)
            and the values are the initializer classes used to initialize
            these data arrays upon release

        :param substance='oil_conservative': Type of oil spilled.
            If this is a string, or an oillibrary.models.Oil object, then
            create gnome.spill.OilProps(oil) object. If this is a
            gnome.spill.OilProps object, then simply instance oil_props
            variable to it: self.oil_props = oil
        :type substance: either str, or oillibrary.models.Oil object or
                         gnome.spill.OilProps
        :param density=None: Allow user to set oil density directly.
        :param density_units='kg/m^3: Only used if a density is input.
        '''
        self.initializers = initializers
        if isinstance(substance, basestring):
            # leave for now to preserve tests
            self.substance = OilProps(substance)
        else:
            # assume object passed in is an OilProps object
            self.substance = substance

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'initializers={0.initializers}, '
                'substance={0.substance!r}'
                ')'.format(self))

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        '''
        call all initializers. This will set the initial values for all
        data_arrays.
        '''
        if num_new_particles > 0:
            for key, i in self.initializers.iteritems():
                if key in data_arrays:
                    i.initialize(num_new_particles, spill, data_arrays,
                                 self.substance)

    def to_dict(self):
        """
        call the to_dict method on each object in the initializers dict. Store
        results in dict and return.

        todo: the standard to_dict doesn't seem to fit well in this case. It
        works but perhaps can/should be revisited to make it simpler
        """
        dict_ = super(ElementType, self).to_dict()

        init = {}
        for key, val in dict_['initializers'].iteritems():
            init[key] = val.to_dict()

        dict_['initializers'] = init
        return dict_

    def initializers_to_dict(self):
        'just return a deepcopy of the initializers'
        return copy.deepcopy(self.initializers)

    def serialize(self, json_='webapi'):
        """
        serialize each object in 'initializers' dict, then add it to the json
        for the ElementType object.

        Note: the to_dict() method returns a dict of initializers as well;
        however, the schemas associated with the initializers are dynamic
        (eg initializers that contain a distribution). It is easier to call the
        initializer's serialize() method instead of adding the initializer's
        schemas to the ElementType schema since they are not known ahead of
        time.
        """
        dict_ = self.to_serialize(json_)
        #et_schema = elements_schema.ElementType()
        et_schema = self.__class__._schema()
        et_json_ = et_schema.serialize(dict_)
        s_init = {}

        for i_key, i_val in self.initializers.iteritems():
            s_init[i_key] = i_val.serialize(json_)

        et_json_['initializers'] = s_init
        return et_json_

    @classmethod
    def deserialize(cls, json_):
        """
        deserialize each object in the 'initializers' dict, then add it to
        deserialized ElementType dict
        """
        et_schema = cls._schema()
        dict_ = et_schema.deserialize(json_)
        d_init = {}

        for i_key, i_val in json_['initializers'].iteritems():
            deserial = eval(i_val['obj_type']).deserialize(i_val)

            if json_['json_'] == 'save':
                '''
                If loading from save file, convert the dict_ to new object
                here itself
                '''
                obj = eval(deserial['obj_type']).new_from_dict(deserial)
                d_init[i_key] = obj
            else:
                d_init[i_key] = deserial

        dict_['initializers'] = d_init

        return dict_


def floating(windage_range=(.01, .04), windage_persist=900):
    """
    Helper function returns an ElementType object containing 'windages'
    initializer with user specified windage_range and windage_persist.
    """
    return ElementType({'windages': InitWindages(windage_range,
                                                 windage_persist)})


def plume(distribution_type='droplet_size',
          distribution='weibull',
          windage_range=(.01, .04),
          windage_persist=900,
          substance_name='oil_conservative',
          density=None,
          density_units='kg/m^3',
          **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages'
    initializer with user specified parameters for distribution.

    See below docs for details on the parameters.

    :param str distribution_type: default ='droplet_size'
                                  available options:
                                  - 'droplet_size': Droplet size is samples
                                                    from the specified
                                                    distribution. Rise velocity
                                                    is calculated.
                                  - 'rise_velocity': rise velocity is directly
                                                     sampled from the specified
                                                     distribution. No droplet
                                                     size is computed.
    :param distribution='weibull':
    :param windage_range=(.01, .04):
    :param windage_persist=900:
    :param substance_name='oil_conservative':
    :param density = None:
    :param density_units = 'kg/m^3':
    """
    if density is not None:
        substance = OilPropsFromDensity(density, substance_name, density_units)
    else:
        substance = OilProps(substance_name)

    if distribution_type == 'droplet_size':
        return ElementType({'rise_vel': InitRiseVelFromDropletSizeFromDist(distribution=distribution,
                                                                           **kwargs),
                            'windages': InitWindages(windage_range,
                                                     windage_persist),
                            'mass': InitMassFromTotalMass()},
                           substance)
    elif distribution_type == 'rise_velocity':
        return ElementType({'rise_vel': InitRiseVelFromDist(distribution=distribution,
                                                            **kwargs),
                            'windages': InitWindages(windage_range,
                                                     windage_persist),
                            'mass': InitMassFromTotalMass()},
                           substance)


## Add docstring from called classes

plume.__doc__ += ("\nDocumentation of OilPropsFromDensity:\n" +
                   OilPropsFromDensity.__init__.__doc__ +
                   "\nDocumentation of InitRiseVelFromDropletSizeFromDist:\n" +
                   InitRiseVelFromDropletSizeFromDist.__init__.__doc__ +
                   "\nDocumentation of InitRiseVelFromDist:\n" +
                   InitRiseVelFromDist.__init__.__doc__ +
                   "\nDocumentation of InitWindages:\n" +
                   InitWindages.__init__.__doc__ +
                   "\nDocumentation of InitMassFromVolume:\n" +
                   InitMassFromTotalMass.__init__.__doc__
                   )


def plume_from_model(distribution_type='droplet_size',
                     distribution='weibull',
                     windage_range=(.01, .04),
                     windage_persist=900,
                     **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages'
    initializer with user specified parameters for distribution.
    """
    if distribution_type == 'droplet_size':
        return ElementType({'rise_vel': InitRiseVelFromDropletSizeFromDist(distribution=distribution,
                                                 **kwargs),
                            'windages': InitWindages(windage_range,
                                                     windage_persist),
                            'mass': InitMassFromPlume()})
    elif distribution_type == 'rise_velocity':
        return ElementType({'rise_vel': InitRiseVelFromDist(distribution=distribution,
                                                 **kwargs),
                            'windages': InitWindages(windage_range,
                                                     windage_persist),
                            'mass': InitMassFromPlume()})
