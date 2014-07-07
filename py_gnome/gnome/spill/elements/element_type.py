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

import gnome    # required by new_from_dict
from gnome.utilities.serializable import Serializable
from .initializers import (InitRiseVelFromDropletSizeFromDist,
                           InitRiseVelFromDist,
                           InitWindages,
                           InitMassFromTotalMass,
                           InitMassFromPlume)
from gnome.db.oil_library.oil_props import (OilProps, OilPropsFromDensity)

from gnome.persist import base_schema

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


def floating_mass(windage_range=(.01, .04), windage_persist=900):
    """
    Helper function returns an ElementType object containing 'windages'
    initializer with user specified windage_range and windage_persist.
    """
    return ElementType({'windages': InitWindages(windage_range,
                                                 windage_persist),
                        'mass': InitMassFromTotalMass()})


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
