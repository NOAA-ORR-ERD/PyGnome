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
from math import exp, log

import gnome    # required by new_from_dict
from gnome.utilities.serializable import Serializable
from gnome.array_types import num_oil_components, reset_to_defaults
from .initializers import (InitRiseVelFromDropletSizeFromDist,
                           InitRiseVelFromDist,
                           InitWindages,
                           InitMassFromSpillAmount,
                           InitArraysFromOilProps,
                           InitMassFromPlume)
from gnome.environment import water, atmos
from oil_library import get_oil

from gnome.persist import base_schema

""" ElementType classes"""


def vapor_pressure(bp):
    '''
    water_temp and boiling point units are Kelvin
    returns the vapor_pressure in SI units (Pascals)
    '''
    D_Zb = 0.97
    R_cal = 1.987  # calories

    D_S = 8.75 + 1.987 * log(bp)
    C_2i = 0.19 * bp - 18

    var = 1. / (bp - C_2i) - 1. / (water['temperature'] - C_2i)
    ln_Pi_Po = D_S * (bp - C_2i) ** 2 / (D_Zb * R_cal * bp) * var
    Pi = exp(ln_Pi_Po) * atmos['pressure']

    return Pi


class ElementType(Serializable):
    _state = copy.deepcopy(Serializable._state)
    _state.add(save=['initializers'], update=['initializers'])
    _schema = base_schema.ObjType

    def __init__(self, initializers=[], substance='oil_conservative'):
        '''
        Define initializers for the type of elements

        :param iterbale initializers: a list/tuple of initializer classes used
            to initialize these data arrays upon release. If this is not an
            iterable, then just append 'initializer' to list of initializers
            assuming it is just a single initializer object

        :param substance='oil_conservative': Type of oil spilled. If this is a
            string, then use get_oil to get the OilProps object, else assume it
            is an OilProps object
        :type substance: str or OilProps
        :param density=None: Allow user to set oil density directly.
        :param density_units='kg/m^3: Only used if a density is input.
        '''
        self.initializers = []
        try:
            self.initializers.extend(initializers)
        except TypeError:
            # initializers is not an iterable so assume its an object and just
            # append it to list
            self.initializers.append(initializers)

        if isinstance(substance, basestring):
            # leave for now to preserve tests
            self.substance = get_oil(substance, 2)
        else:
            self.substance = substance

        self.substance.temperature = water['temperature']
        if self.substance.num_components != num_oil_components:
            reset_to_defaults()

        # for now add vapor_pressure here
        self.vapor_pressure = [vapor_pressure(bp)
                               for bp in self.substance.boiling_point]

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'initializers={0.initializers}, '
                'substance={0.substance!r}'
                ')'.format(self))

    @property
    def array_types(self):
        '''
        compile/return dict of array_types set by all initializers contained
        by ElementType object
        '''
        at = {}
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
                # If a mover is using an initializers, the data_arrays will
                # contain all
                p_key = i.array_types.keys()[0]
                if p_key in data_arrays:
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
        et_schema = self.__class__._schema()
        et_json_ = et_schema.serialize(dict_)
        #s_init = {}
        s_init = []

        for i_val in self.initializers:
            s_init.append(i_val.serialize(json_))
            #s_init[i_key] = i_val.serialize(json_)

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
        #d_init = {}
        d_init = []

        for i_val in json_['initializers']:
            deserial = eval(i_val['obj_type']).deserialize(i_val)

            if json_['json_'] == 'save':
                '''
                If loading from save file, convert the dict_ to new object
                here itself
                '''
                obj = eval(deserial['obj_type']).new_from_dict(deserial)
                #d_init[i_key] = obj
                d_init.append(obj)
            else:
                #d_init[i_key] = deserial
                d_init.append(deserial)

        dict_['initializers'] = d_init

        return dict_


def floating(windage_range=(.01, .04),
             windage_persist=900,
             substance=None):
    """
    Helper function returns an ElementType object containing 'windages'
    initializer with user specified windage_range and windage_persist.
    """
    init = [InitWindages(windage_range, windage_persist)]
    if substance:
        ElementType(init, substance)
    else:
        return ElementType(init)


def floating_mass(windage_range=(.01, .04),
                  windage_persist=900,
                  substance=None):
    """
    Helper function returns an ElementType object containing 'windages'
    initializer with user specified windage_range and windage_persist.
    """
    init = [InitWindages(windage_range, windage_persist),
            InitMassFromSpillAmount()]
    if substance:
        return ElementType(init, substance)
    else:
        return ElementType(init)


def floating_weathering(windage_range=(.01, .04),
                        windage_persist=900,
                        substance=None):
    '''
    Use InitArraysFromOilProps()
    '''
    init = [InitWindages(windage_range, windage_persist),
            InitArraysFromOilProps()]
    if substance:
        return ElementType(init, substance)
    else:
        return ElementType(init)


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
        return ElementType([InitRiseVelFromDropletSizeFromDist(
                                distribution=distribution, **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromSpillAmount()],
                           substance)
    elif distribution_type == 'rise_velocity':
        return ElementType([InitRiseVelFromDist(distribution=distribution,
                                                **kwargs),
                            InitWindages(windage_range, windage_persist),
                            InitMassFromSpillAmount()],
                           substance)


## Add docstring from called classes

plume.__doc__ += ("\nDocumentation of InitRiseVelFromDropletSizeFromDist:\n" +
                   InitRiseVelFromDropletSizeFromDist.__init__.__doc__ +
                   "\nDocumentation of InitRiseVelFromDist:\n" +
                   InitRiseVelFromDist.__init__.__doc__ +
                   "\nDocumentation of InitWindages:\n" +
                   InitWindages.__init__.__doc__ +
                   "\nDocumentation of InitMassFromVolume:\n" +
                   InitMassFromSpillAmount.__init__.__doc__
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
