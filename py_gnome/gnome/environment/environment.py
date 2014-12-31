"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy

from colander import SchemaNode, Float, MappingSchema, drop, String, OneOf
import unit_conversion as uc

from gnome.utilities import serializable
from gnome.persist import base_schema

from .. import _valid_units


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
_valid_temp_units = _valid_units('Temperature')
_valid_dist_units = _valid_units('Length')
_valid_kvis_units = _valid_units('Kinematic Viscosity')
_valid_density_units = _valid_units('Density')
_valid_salinity_units = ('psu',)
_valid_sediment_units = ('mg/l',)


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
                          validator=OneOf(_valid_sediment_units))

    # wave height and fetch have distance units
    wave_height = SchemaNode(String(),
                             description='SI units for distance',
                             validator=OneOf(_valid_dist_units))

    fetch = SchemaNode(String(),
                       description='SI units for distance',
                       validator=OneOf(_valid_dist_units))
    kinematic_viscosity = SchemaNode(String(),
                                     description='SI units for viscosity',
                                     validator=OneOf(_valid_kvis_units))
    density = SchemaNode(String(),
                         description='SI units for density',
                         validator=OneOf(_valid_density_units))


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
               serializable.Field('wave_height', update=True, save=True),
               serializable.Field('density', update=True, save=True),
               serializable.Field('kinematic_viscosity', update=True,
                                  save=True)]

    _schema = WaterSchema

    _units_type = {'temperature': ('temperature', _valid_temp_units),
                   'salinity': ('salinity', _valid_salinity_units),
                   'sediment': ('sediment', _valid_sediment_units),
                   'wave_height': ('length', _valid_dist_units),
                   'fetch': ('length', _valid_dist_units),
                   'kinematic_viscosity': ('kinematic viscosity',
                                           _valid_kvis_units),
                   'density': ('density', _valid_density_units),
                   }

    def __init__(self,
                 temperature=311.15,
                 salinity=35.0,
                 sediment=None,
                 wave_height=None,
                 fetch=None,
                 name='Water'):
        '''
        Assume units are SI for all properties. 'units' attribute assumes SI
        by default. This can be changed, but initialization takes SI.
        '''
        # define properties in SI units
        # ask if we want unit conversion implemented here?
        self.units = {'temperature': 'K',
                      'salinity': 'psu',
                      'sediment': 'mg/l',  # do we need SI here?
                      'wave_height': 'm',
                      'fetch': 'm',
                      'density': 'kg/m^3',
                      'kinematic_viscosity': 'm^2/s'}
        self.temperature = temperature
        self.salinity = salinity
        self.sediment = sediment
        self.wave_height = wave_height
        self.fetch = fetch
        self.density = 997
        self.kinematic_viscosity = 0.000001
        self.name = name

    def __repr__(self):
        info = ("{0.__class__.__module__}.{0.__class__.__name__}"
                "(temperature={0.temperature},"
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
            # Note: salinity and sediment only have one units since we don't
            # have any conversions for them in unit_conversion yet - revisit this per
            # requirements
            return val

        if unit in self._units_type[attr][1]:
            return uc.convert(self._units_type[attr][0], self.units[attr],
                              unit, val)
        else:
            # log to file if we have logger
            raise uc.InvalidUnitError((unit, self._units_type[attr][0]))

    def set(self, attr, value, unit):
        '''
        provide a corresponding set method that requires value and units
        The attributes can be directly set. This function just sets the
        desired property and also updates the units dict
        '''
        if unit not in self._units_type[attr][1]:
            raise uc.InvalidUnitError((unit, self._units_type[attr][0]))

        setattr(self, attr, value)
        self.units[attr] = unit
