"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy
from itertools import chain

from colander import SchemaNode, Float, MappingSchema, drop, String, OneOf
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


# define valid units at module scope because the Schema and Object both use it
_valid_temp_units = uc.GetUnitNames('Temperature')
_valid_temp_units.extend(
    chain(*[val[1] for val in
            uc.ConvertDataUnits['Temperature'].values()]))

_valid_ciw_units = uc.GetUnitNames('Concentration In Water')
_valid_ciw_units.extend(
    chain(*[val[1] for val in
            uc.ConvertDataUnits['Concentration In Water'].values()]))

_valid_dist_units = uc.GetUnitNames('Length')
_valid_dist_units.extend(
    chain(*[val[1] for val in
            uc.ConvertDataUnits['Length'].values()]))

_valid_salinity_units = ['psu']


class UnitsSchema(MappingSchema):
    temperature = SchemaNode(String(),
                             description='SI units for temp',
                             validator=OneOf(_valid_temp_units))

    # for now salinity only has one units
    salinity = SchemaNode(String(),
                          description='SI units for salinity',
                          validator=OneOf(_valid_salinity_units))

    # sediment load units? Concentration In Water?
    sediment = SchemaNode(String(),
                          description='SI units for density',
                          validator=OneOf(_valid_ciw_units))

    # wave height and fetch have distance units
    wave_height = SchemaNode(String(),
                             description='SI units for distance',
                             validator=OneOf(_valid_dist_units))

    fetch = SchemaNode(String(),
                       description='SI units for distance',
                       validator=OneOf(_valid_dist_units))


class WaterSchema(base_schema.ObjType):
    'Colander Schema for Conditions object'
    units = UnitsSchema()
    temperature = SchemaNode(Float())
    salinity = SchemaNode(Float())
    sediment = SchemaNode(Float(), missing=drop)
    wave_height = SchemaNode(Float(), missing=drop)
    fetch = SchemaNode(Float(), missing=drop)


class Water(Environment, serializable.Serializable):
    '''
    Define the environmental conditions for a spill, like water_temperature,
    atmos_pressure (most likely a constant)

    Defined in a Serializable class since user will need to set/get some of
    these properties through the client
    '''
    _state = copy.deepcopy(Environment._state)
    _state += [serializable.Field('units', update=True, save=True),
               serializable.Field('temperature', update=True, save=True),
               serializable.Field('salinity', update=True, save=True),
               serializable.Field('sediment', update=True, save=True),
               serializable.Field('fetch', update=True, save=True),
               serializable.Field('wave_height', update=True, save=True)]

    _schema = WaterSchema

    _units_type = {'temperature': ('temperature', _valid_temp_units),
                   'salinity': ('salinity', _valid_salinity_units),
                   'sediment': ('Concentration In Water', _valid_ciw_units),
                   'wave_height': ('length', _valid_dist_units),
                   'fetch': ('length', _valid_dist_units)}

    def __init__(self,
                 temperature=311.15,
                 salinity=35.0,
                 name='WaterProperties'):
        '''
        Assume units are SI
        '''
        # define properties in SI units
        # ask if we want unit conversion implemented here?
        self.units = {'temperature': 'K',
                      'salinity': 'psu',
                      'sediment': 'kg/m^3',  # double check these
                      'wave_height': 'm',
                      'fetch': 'm'}
        self.temperature = temperature
        self.salinity = salinity
        self.sediment = None
        self.wave_height = None
        self.fetch = None
        self.name = name

    def __repr__(self):
        info = ("{0.__class__.__module__}.{0.__class__.__name__}"
                "(temperature={0.temperature}",
                " salinity={0.salinity})").format(self)
        return info

    __str__ = __repr__

    def get(self, attr, unit=None):
        '''
        return value in desired unit. If None, then return the value without
        any conversion. The unit are given in 'unit' attribute. Can also
        get the values directly from 'water', 'atmos', 'constant' dicts - just
        be sure the unit are as desired

        .. note:: unit_conversion does not contain a conversion for 'pressure'
        Need to add this at some point for completeness
        '''
        val = getattr(self, attr)
        if unit is None or unit == self.units[attr]:
            # Note: no conversion for salinity yet so just retrun the one value
            return val

        if unit in self._units_type[attr][1]:
            return uc.convert(self._units_type[attr][0], self.units[attr],
                              unit, val)
        else:
            # log to file if we have logger
            raise uc.InvalidUnitError(unit, self._units_type[attr][0])

    def set(self, attr, value, unit):
        '''
        provide a corresponding set method that requires value and units
        The attributes can be directly set. This function just sets the
        desired property and also updates the units dict
        '''
        if unit not in self._units_type[attr][1]:
            raise uc.InvalidUnitError(unit, self._units_type[attr][0])

        setattr(self, attr, value)
        self.units[attr] = unit
