#!/usr/bin/env python
'''
Types of elements that a spill can expect
These are properties that are spill specific like:

  'floating' element_types would contain windage_range, windage_persist
  'subsurface_dist' element_types would contain rise velocity distribution info
  'nonweathering' element_types would set use_droplet_size flag to False
  'weathering' element_types would use droplet_size, densities, mass?

Note: An ElementType needs a bunch of initializers -- but that is an
      implementation detail, so the ElementType API exposes access to the
      initializers.
'''

import copy

import unit_conversion as uc
from colander import SchemaNode, SequenceSchema, Float
from gnome.persist import base_schema

from .substance import NonWeatheringSubstance
from .initializers import (InitRiseVelFromDropletSizeFromDist,
                           InitRiseVelFromDist,
                           InitWindages,
                           InitMassFromPlume)
from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.spill.elements.initializers import (InitWindagesSchema,
                                               DistributionBaseSchema)


class ElementTypeSchema(base_schema.ObjTypeSchema):
    def __init__(self, unknown='preserve', *args, **kwargs):
        super(ElementTypeSchema, self).__init__(*args, **kwargs)
        self.typ = base_schema.ObjType('preserve')

    initializers = SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[InitWindagesSchema,
                                DistributionBaseSchema
                                ]
        ),
        save=True, update=True, save_reference=True
    )
    standard_density = SchemaNode(
        Float(), read_only=True
    )


class ElementType(GnomeId):

    _schema = ElementTypeSchema

    def __init__(self, initializers=[], substance=None, standard_density=1000, *args, **kwargs):
        '''
        Define initializers for the type of elements.
        The default element_type has a substance with density of water
        (1000 kg/m^3). This is labeled as 'oil_conservaitve', same as in
        original gnome. This is currently one of the mock ("fake") oil objects,
        used primarily to help integrate weathering processes. It doesn't mean
        weathering is off - if there are no weatherers, then oil doesn't
        weather.

        :param iterable initializers: a list/tuple of initializer classes used
            to initialize these data arrays upon release. If this is not an
            iterable, then just append 'initializer' to list of initializers
            assuming it is just a single initializer object
        :param substance=None: Type of oil spilled. If this is a
            string, then use get_oil_props to get the OilProps object, else
            assume it is an OilProps object. If it is None, then assume there
            is no weathering.
        :type substance: str or OilProps

        '''
        super(ElementType, self).__init__(*args, **kwargs)
        from oil_library import get_oil_props
        self.get_oil_props = get_oil_props

        self.initializers = []
        try:
            self.initializers.extend(initializers)
        except TypeError:
            # initializers is not an iterable so assume its an object and just
            # append it to list
            self.initializers.append(initializers)


        self._substance = None
        if substance is not None:
            self.substance = substance

        self.logger.debug(self._pid + 'constructed element_type: ' +
                          self.__class__.__name__)

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'initializers={0.initializers}, '
                'substance={0.substance!r}'
                ')'.format(self))

    # properties for attributes the need to be pulled from initializers
    @property
    def windage_range(self):
        for initr in self.initializers:
            try:
                return getattr(initr, 'windage_range')
            except AttributeError:
                pass
        msg = 'windage_range attribute does not exist any initializers'

        self.logger.warning(msg)
        raise AttributeError(msg)

    @windage_range.setter
    def windage_range(self, wr):
        print self.initializers
        for initr in self.initializers:
            print "initr:"
            if hasattr(initr, "windage_range"):
                print "setting windage_range"
                initr.windage_range = wr
                return None
        msg = "can't set windage_range: no initializer has it"

        self.logger.warning(msg)
        raise AttributeError(msg)

    @property
    def windage_persist(self):
        for initr in self.initializers:
            try:
                return getattr(initr, 'windage_persist')
            except AttributeError:
                pass
        msg = 'windage_persist attribute does not exist any initializers'

        self.logger.warning(msg)
        raise AttributeError(msg)

    @windage_persist.setter
    def windage_persist(self, wp):
        print self.initializers
        for initr in self.initializers:
            print "initr:"
            if hasattr(initr, "windage_persist"):
                print "setting windage_persist"
                initr.windage_persist = wp
                return None
        msg = "can't set windage_persist: no initializer has it"

        self.logger.warning(msg)
        raise AttributeError(msg)

    @property
    def standard_density(self):
        '''
            Get the substance's standard density if it exists, otherwise
            default to 1000.0 kg/m^3.
            Any valid substance object needs to have a property named
            standard_density.
        '''
        if (self.substance is not None and
                hasattr(self.substance, 'standard_density')):
            return self.substance.standard_density
        else:
            return 1000.0

    def contains_object(self, obj_id):
        for o in self.initializers:
            if obj_id == o.id:
                return True

            if (hasattr(o, 'contains_object') and
                    o.contains_object(obj_id)):
                return True

        return False

    @classmethod
    def new_from_dict(cls, dict_):
        return super(ElementType, cls).new_from_dict(dict_)

    def to_dict(self, json_=None):
        dict_ = super(ElementType, self).to_dict(json_=json_)
        #append substance because no good schema exists for it
        if json_ != 'save':
            dict_['substance'] = self.substance_to_dict()
        else:
            if self.substance is not None:
                dict_['substance'] = self.substance_to_dict()['name']
        return dict_

    def serialize(self):
        ser = GnomeId.serialize(self)
        ser['substance'] = self.to_dict()['substance']
        return ser

    @classmethod
    def deserialize(cls, json_):
        if 'substance' in json_:
            sub = json_['substance']
        obj = super(ElementType, cls).deserialize(json_)
        obj.substance = sub['name']
        return obj

    @classmethod
    def load(cls, saveloc='.', filename=None, refs=None):
        return super(ElementType, cls).load(saveloc=saveloc, filename=filename, refs=refs)

    def substance_to_dict(self):
        '''
        Call the tojson() method on substance

        An Oil object that has been queried from the database
        contains a lot of unnecessary relationships that we do not
        want to represent in our JSON output,

        So we prune them by first constructing an Oil object from the
        JSON payload of the queried Oil object.

        This creates an Oil object in memory that does not have any
        database links. Then output the JSON from the unlinked object.
        '''
        if self._substance is not None:
            return self._prune_substance(self._substance.tojson())


    def _prune_substance(self, substance_json):
        '''
            Whether saving to a savefile or outputting to the web client,
            the id of the substance objects is not necessary, and in fact
            not even wanted.

            Except for the main oil ID from the database.
        '''
        del substance_json['imported_record_id']
        del substance_json['estimated_id']

        for attr in ('kvis', 'densities', 'cuts',
                     'molecular_weights',
                     'sara_densities', 'sara_fractions'):
            for item in substance_json[attr]:
                for sub_item in ('id', 'oil_id', 'imported_record_id'):
                    if sub_item in item:
                        del item[sub_item]

        return substance_json

    @property
    def substance(self):
        return self._substance

    @substance.setter
    def substance(self, val):
        '''
        first try to use get_oil_props using 'val'. If this fails, then assume
        user has provided a valid OilProps object and use it as is
        '''
        try:
            self._substance = self.get_oil_props(val)
        except Exception:
            if isinstance(val, basestring):
                raise

            self.logger.info('Failed to get_oil_props for {0}. Use as is '
                             'assuming has OilProps interface'.format(val))
            self._substance = val

    @property
    def array_types(self):
        '''
        compile/return dict of array_types set by all initializers contained
        by ElementType object
        '''
        at = set()
        for init in self.initializers:
            at.update(init.array_types)

        return at

    def set_newparticle_values(self, num_new_particles, spill, data_arrays):
        '''
        call all initializers. This will set the initial values for all
        data_arrays.
        '''
        if num_new_particles > 0:
            for i in self.initializers:
                # looks like issubset() looks at data_arrays.keys()
                if i.array_types.issubset(data_arrays):
                    i.initialize(num_new_particles, spill, data_arrays,
                                 self.substance)


def floating(windage_range=(.01, .04),
             windage_persist=900,
             substance=None):
    """
    Helper function returns an ElementType object containing following
    initializers:

    1. InitWindages(): for initializing 'windages' with user specified
    windage_range and windage_persist.

    :param substance='oil_conservative': Type of oil spilled. Passed onto
        ElementType constructor
    :type substance: str or OilProps
    """
    init = [InitWindages(windage_range, windage_persist)]
    return ElementType(init, substance)


def plume(distribution_type='droplet_size',
          distribution=None,
          windage_range=(.01, .04),
          windage_persist=900,
          substance_name=None,
          density=None,
          density_units='kg/m^3',
          **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages' initialized with user specified parameters for distribution.

    :param str distribution_type='droplet_size': type of distribution

    :param gnome.utilities.distributions distribution=None:

    :param windage_range=(.01, .04): minimum and maximum windage
    :type windage_range: tuple-of-floats

    :param int windage_persist=900: persistance of windage in seconds

    :param str substance_name=None:

    :param float density = None:

    :param str density_units='kg/m^3':

    Distribution type Available options:

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

    New distribution classes could be made.  The only requirement is they
    need to have a ``set_values()`` method which accepts a NumPy array.
    (presumably, this function will also modify the array in some way)

    .. note:: substance_name or density must be provided

    """
    from oil_library import get_oil_props

    # Add docstring from called classes
    # Note: following gives sphinx warnings on build, ignore for now.

    plume.__doc__ += ("\nInitRiseVelFromDropletSizeFromDist Documentation:\n" +
                      InitRiseVelFromDropletSizeFromDist.__init__.__doc__ +
                      "\nInitRiseVelFromDist Documentation:\n" +
                      InitRiseVelFromDist.__init__.__doc__ +
                      "\nInitWindages Documentation:\n" +
                      InitWindages.__init__.__doc__
                      )

    if density is not None:
        # Assume density is at 15 C
        substance = NonWeatheringSubstance(standard_density=density)
    elif substance_name is not None:
        # model 2 cuts if fake oil
        substance = get_oil_props(substance_name, 2)
    else:
        raise ValueError('plume substance density and/or name '
                         'must be provided')

    if distribution_type == 'droplet_size':
        return ElementType([InitRiseVelFromDropletSizeFromDist(
                            distribution=distribution, **kwargs),
                            InitWindages(windage_range, windage_persist)],
                           substance)
    elif distribution_type == 'rise_velocity':
        return ElementType([InitRiseVelFromDist(distribution=distribution,
                                                **kwargs),
                            InitWindages(windage_range, windage_persist)],
                           substance)
    else:
        raise TypeError('distribution_type must be either droplet_size or '
                        'rise_velocity')


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
        return ElementType([InitRiseVelFromDropletSizeFromDist(
                            distribution=distribution, **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromPlume()])
    elif distribution_type == 'rise_velocity':
        return ElementType([InitRiseVelFromDist(distribution=distribution,
                                                **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromPlume()])
