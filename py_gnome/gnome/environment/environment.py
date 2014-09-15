"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy

from colander import SchemaNode, Float, MappingSchema, String, drop
from hazpy import unit_conversion as uc

from gnome.utilities import serializable
from gnome.persist import base_schema


class Environment(object):
    """
    A base class for all classes in environment module

    This is primarily to define a dtype such that the OrderedCollection
    defined in the Model object requires it.
    """
    _state = copy.deepcopy(serializable.Serializable._state)

    def __init__(self, name=None):
        '''
        base class for environment objects

        :param name=None:
        '''
        if name:
            self.name = name


class UnitsSchema(MappingSchema):
    temperature = SchemaNode(String(),
                             description='SI units for temp')
    pressure = SchemaNode(String(),
                          description='SI units for pressure')


class WaterSchema(MappingSchema):
    temperature = SchemaNode(Float(),
                             description='water temp in SI units, Kelvin')


class AtmosphereSchema(MappingSchema):
    pressure = SchemaNode(Float(),
                          description='atmospheric pressure in SI units, Pa')


class ConditionsSchema(base_schema.ObjType):
    'Colander Schema for Conditions object'
    units = UnitsSchema()
    water = WaterSchema()
    atmos = AtmosphereSchema()


class Conditions(Environment, serializable.Serializable):
    '''
    Define the environmental conditions for a spill, like water_temperature,
    atmos_pressure (most likely a constant)

    Defined in a Serializable class since user will need to set/get some of
    these properties through the client
    '''
    _state = copy.deepcopy(Environment._state)
    _state += [serializable.Field('units', update=True, save=True),
               serializable.Field('water', update=True, save=True),
               serializable.Field('atmos', update=True, save=True)]

    _schema = ConditionsSchema

    def __init__(self,
                 water_temp=311.15,
                 name='conditions'):
        '''
        Assume units are SI
        '''
        # define global environmental properties in SI units
        # ask if we want unit conversion implemented here?
        self.units = {'temperature': 'K',
                      'pressure': 'Pa'}
        self.water = {'temperature': water_temp}
        self.atmos = {'pressure': 101325.0}
        self.name = name

    def __repr__(self):
        info = ("{0.__class__.__module__}.{0.__class__.__name__}"
                "(water_temp={0.water[temperature]})").format(self)
        return info

    __str__ = __repr__

    def get(self, attr, key, units=None):
        '''
        return value in desired units. If None, then return the value without
        any conversion. The units are given in 'units' attribute. Can also
        get the values directly from 'water', 'atmos', 'constant' dicts - just
        be sure the units are as desired

        .. note:: unit_conversion does not contain a conversion for 'pressure'
        Need to add this at some point for completeness
        '''
        val = getattr(self, attr)[key]
        if units is None:
            return val
        else:
            if (attr == 'pressure' and units is not None):
                raise NotImplementedError("Unit converter needs a conversion "
                                          "for pressure")

            return uc.convert(key, self.units[key], units, val)
