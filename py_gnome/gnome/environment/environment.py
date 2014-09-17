"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy
from itertools import chain

from colander import SchemaNode, Float, MappingSchema, String, drop, OneOf
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
    _valid_temp_units = uc.GetUnitNames('Temperature')
    _valid_temp_units.extend(
        chain(*[val[1] for val in
                uc.ConvertDataUnits['Temperature'].values()]))
    temperature = SchemaNode(String(),
                             description='SI units for temp',
                             validator=OneOf(_valid_temp_units))

    # for now salinity only has one units
    salinity = SchemaNode(String(),
                          description='SI units for pressure',
                          validator=OneOf(['psu']))


class WaterPropertiesSchema(base_schema.ObjType):
    'Colander Schema for Conditions object'
    units = UnitsSchema()
    temperature = SchemaNode(Float())
    salinity = SchemaNode(Float())


class WaterProperties(Environment, serializable.Serializable):
    '''
    Define the environmental conditions for a spill, like water_temperature,
    atmos_pressure (most likely a constant)

    Defined in a Serializable class since user will need to set/get some of
    these properties through the client
    '''
    _state = copy.deepcopy(Environment._state)
    _state += [serializable.Field('units', update=True, save=True),
               serializable.Field('temperature', update=True, save=True),
               serializable.Field('salinity', update=True, save=True)]

    _schema = WaterPropertiesSchema

    def __init__(self,
                 temperature=311.15,
                 name='WaterProperties'):
        '''
        Assume units are SI
        '''
        # define properties in SI units
        # ask if we want unit conversion implemented here?
        self.units = {'temperature': 'K',
                      'salinity': 'psu'}
        self.temperature = temperature
        self.salinity = 0.0
        self.name = name

    def __repr__(self):
        info = ("{0.__class__.__module__}.{0.__class__.__name__}"
                "(temperature={0.temperature})").format(self)
        return info

    __str__ = __repr__

    def get(self, attr, units=None):
        '''
        return value in desired units. If None, then return the value without
        any conversion. The units are given in 'units' attribute. Can also
        get the values directly from 'water', 'atmos', 'constant' dicts - just
        be sure the units are as desired

        .. note:: unit_conversion does not contain a conversion for 'pressure'
        Need to add this at some point for completeness
        '''
        val = getattr(self, attr)
        if units is None or units == self.units[attr]:
            return val
        else:
            return uc.convert(attr, self.units[attr], units, val)
