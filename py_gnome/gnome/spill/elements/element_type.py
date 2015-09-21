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

from gnome.utilities.serializable import Serializable, Field
from .initializers import (InitRiseVelFromDropletSizeFromDist,
                           InitRiseVelFromDist,
                           InitWindages,
                           InitMassFromPlume)
from oil_library import get_oil_props
from oil_library import build_oil_props
from gnome.persist import base_schema, class_from_objtype
import unit_conversion as uc


class ElementType(Serializable):
    _state = copy.deepcopy(Serializable._state)
    _state.add(save=['initializers'],
               update=['initializers'])
    _state += Field('substance', save=True, update=True, test_for_eq=False)
    _schema = base_schema.ObjType

    def __init__(self, initializers=[], substance=None):
        '''
        Define initializers for the type of elements.
        The default element_type has a substance with density of water
        (1000 kg/m^3). This is labeled as 'oil_conservaitve', same as in
        original gnome. This is currently one of the mock ("fake") oil objects,
        used primarily to help integrate weathering processes. It doesn't mean
        weathering is off - if there are no weatherers, then oil doesn't
        weather.

        :param iterbale initializers: a list/tuple of initializer classes used
            to initialize these data arrays upon release. If this is not an
            iterable, then just append 'initializer' to list of initializers
            assuming it is just a single initializer object
        :param substance=None: Type of oil spilled. If this is a
            string, then use get_oil_props to get the OilProps object, else
            assume it is an OilProps object. If it is None, then assume there
            is no weathering.
        :type substance: str or OilProps

        '''
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

    def contains_object(self, obj_id):
        for o in self.initializers:
            if obj_id == o.id:
                return True

            if (hasattr(o, 'contains_object') and
                    o.contains_object(obj_id)):
                return True

        return False

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
        '''
        del substance_json['id']
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
            self._substance = get_oil_props(val)
        except:
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

    def to_dict(self):
        """
        call the to_dict method on each object in the initializers dict. Store
        results in dict and return.

        todo: the standard to_dict doesn't seem to fit well in this case. It
        works but perhaps can/should be revisited to make it simpler
        """
        dict_ = super(ElementType, self).to_dict()

        init = []
        for val in dict_['initializers']:
            init.append(val.to_dict())

        dict_['initializers'] = init
        return dict_

    def initializers_to_dict(self):
        'just return the initializers'
        return self.initializers

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
        et_schema = self.__class__._schema()
        et_json_ = et_schema.serialize(dict_)
        s_init = []

        for i_val in self.initializers:
            s_init.append(i_val.serialize(json_))

        et_json_['initializers'] = s_init
        if 'substance' in dict_:
            et_json_['substance'] = dict_['substance']

        return et_json_

    @classmethod
    def deserialize(cls, json_):
        """
        deserialize each object in the 'initializers' dict, then add it to
        deserialized ElementType dict

        We also need to accept sparse json objects, in which case we will
        not treat them, but just send them back.
        """
        if not cls.is_sparse(json_):
            et_schema = cls._schema()

            # replace substance with just the oil record ID for now since
            # we don't have a way to construct to object fromjson()
            dict_ = et_schema.deserialize(json_)

            if 'substance' in json_ and json_['substance'] is not {}:
                # no colander validation for oil object
                dict_['substance'] = json_['substance']

            d_init = []

            for i_val in json_['initializers']:
                i_cls = class_from_objtype(i_val['obj_type'])
                deserial = i_cls.deserialize(i_val)

                if json_['json_'] == 'save':
                    '''
                    If loading from save file, convert the dict_ to new object
                    here itself
                    '''
                    obj = i_cls.new_from_dict(deserial)
                    d_init.append(obj)
                else:
                    d_init.append(deserial)

            dict_['initializers'] = d_init

            return dict_
        else:
            return json_


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
          distribution='weibull',
          windage_range=(.01, .04),
          windage_persist=900,
          substance_name=None,
          density=None,
          density_units='kg/m^3',
          **kwargs):
    """
    Helper function returns an ElementType object containing 'rise_vel'
    and 'windages'
    initializer with user specified parameters for distribution.

    See below docs for details on the parameters.
    
    NOTE: substance_name or density must be provided 

    :param str distribution_type: default 'droplet_size' available options:

        1. 'droplet_size': Droplet size is samples from the specified
        distribution. Rise velocity is calculated.

        2.'rise_velocity': rise velocity is directly sampled from the specified
        distribution. No droplet size is computed.

    :param distribution='weibull':
    :param windage_range=(.01, .04):
    :param windage_persist=900:
#    :param substance_name='oil_conservative':
    :param substance_name=None:
    :param float density = None:
    :param str density_units='kg/m^3':
    """
    if density is not None:
        # Assume density is at 15 K - convert density to api
        api = uc.convert('density', density_units, 'API', density)
        if substance_name is not None:
            substance = build_oil_props({'name':substance_name, 'api': api}, 2)
        else:
            substance = build_oil_props({'api': api}, 2)
    elif substance_name is not None:
        # model 2 cuts if fake oil
        substance = get_oil_props(substance_name, 2)
    else:
        ex = ValueError()
        ex.message = ("plume substance density and/or name must be provided")
        raise ex
        

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


# Add docstring from called classes
# Note: following gives sphinx warnings on build, ignore for now.

plume.__doc__ += ("\nDocumentation of InitRiseVelFromDropletSizeFromDist:\n" +
                  InitRiseVelFromDropletSizeFromDist.__init__.__doc__ +
                  "\nDocumentation of InitRiseVelFromDist:\n" +
                  InitRiseVelFromDist.__init__.__doc__ +
                  "\nDocumentation of InitWindages:\n" +
                  InitWindages.__init__.__doc__
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
        return ElementType([InitRiseVelFromDropletSizeFromDist(
                                distribution=distribution, **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromPlume()])
    elif distribution_type == 'rise_velocity':
        return ElementType([InitRiseVelFromDist(distribution=distribution,
                                                **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromPlume()])
