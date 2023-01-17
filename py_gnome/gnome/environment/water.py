'''
The Water environment object.
The Water object defines the Water conditions for the spill

TEmerature, Salinity, etc.

NOTE: this is simple, constant water conditions -- at some point, we will make it a proper Environment object
'''

from functools import lru_cache
import warnings

import numpy as np

import gsw

import nucos as uc

from gnome import constants
from gnome.utilities.inf_datetime import InfTime, MinusInfTime

from gnome.persist import (base_schema, SchemaNode, MappingSchema, Float,
                           String, drop, OneOf, required)

from .environment import Environment

from .. import _valid_units

# define valid units at module scope because the Schema and Object both use it
_valid_temp_units = _valid_units('Temperature')
_valid_dist_units = _valid_units('Length')
_valid_kvis_units = _valid_units('Kinematic Viscosity')
_valid_density_units = _valid_units('Density')
_valid_salinity_units = ('psu',)
_valid_sediment_units = _valid_units('Concentration In Water')

# New stuff to make a water object more like an environment object


# NOTE: I got a little obsessed with DRY and wrote a fancy decorator system
#       For only three classes, so probably not at all worth it :-) - CHB

class WaterProperty(Environment):
    """
    Base class for making an Environment object for a water property

    These one uses a Water object for the data itself

    Needs to be wrapped with the make_water_property decorator to be functional
    """
    data_start = MinusInfTime()
    data_stop = InfTime()

    def __init__(self, water):
        self.water = water
        super().__init__()

def make_water_property(property, default_unit):

    def wrapper(cls):

        def at(self, points, time, units=default_unit):
            """
            return the temperature at the points and time asked for

            current implementation always returns the same value
            """
            temp = self.water.get(property, unit=units)
            return np.zeros((len(points),)) + temp
        cls.at = at
        return cls

    return wrapper


@make_water_property('temperature', 'K')
class Temperature(WaterProperty):
    pass


@make_water_property('salinity', 'psu')
class Salinity(WaterProperty):
    pass


@make_water_property('sediment', 'kg/m^3')
class Sediment(WaterProperty):
    pass


@make_water_property('wave_height', 'meter')
class WaveHeight(WaterProperty):
    pass


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


class WaterSchema(base_schema.ObjTypeSchema):
    'Colander Schema for Conditions object'
    units = UnitsSchema()
    temperature = SchemaNode(Float())
    salinity = SchemaNode(Float())
    sediment = SchemaNode(Float())
    wave_height = SchemaNode(Float(), missing=None)
    fetch = SchemaNode(Float(), missing=None)


class Water(Environment):
    '''
    Define the environmental conditions for a spill, like water_temperature,

    Single values for all all time and space

    Defined in a Serializable class since user will need to set/get some of
    these properties through the client

    Includes attributes with a Environment interface:

    Temperature
    Salinity
    Sediment
    WaveHeight

    Example: water.Temperature.at(points, time, unit)

    '''
    _ref_as = 'water'
    _field_descr = {
        'units': ('update', 'save'),
        'temperature:': ('update', 'save'),
        'salinity': ('update', 'save'),
        'sediment': ('update', 'save'),
        'fetch': ('update', 'save'),
        'wave_height': ('update', 'save'),
        'density': ('update', 'save'),
        'kinematic_viscosity': ('update', 'save')
    }

    _schema = WaterSchema

    _units_type = {
        'temperature': ('temperature', _valid_temp_units),
        'salinity': ('salinity', _valid_salinity_units),
        'sediment': ('concentration in water', _valid_sediment_units),
        'wave_height': ('length', _valid_dist_units),
        'fetch': ('length', _valid_dist_units),
        'kinematic_viscosity': ('kinematic viscosity', _valid_kvis_units),
        'density': ('density', _valid_density_units),
    }

    # Fixme: SI units are defined by an outside body (that is, SI :-) )
    #        we should not be redefining it locally
    #        We *may* want to have system for "GNOME units" which is almost SI
    #        But if so -- it should be somewhere central. This is NOT Water specific!
    # keep track of valid SI units for properties - these are used for
    # conversion since internal code uses SI units. Don't expect to change
    # these so make it a class level attribute
    _gnome_units = {
        'temperature': 'K',
        'salinity': 'psu',
        'sediment': 'kg/m^3',
        'wave_height': 'm',
        'fetch': 'm',
        'density': 'kg/m^3',
        'kinematic_viscosity': 'm^2/s'
    }

    # Fixme: we need a simple way for (scripting) users to use the units they want to create
    #        a water object.
    def __init__(
            self,
            temperature=300.0,
            salinity=35.0,
            sediment=.005,  # kg/m^3 oceanic default
            wave_height=None,
            fetch=None,
            units=None,
            name='Water',
            **kwargs):
        """
        Assume units are SI for all properties. 'units' attribute assumes SI
        by default. This can be changed, but initialization takes SI.

        Units is dict, with some subset of these keys::

            {
            'temperature': 'K',
            'salinity': 'psu',
            'sediment': 'kg/m^3',
            'wave_height': 'm',
            'fetch': 'm',
            }

        So to set units of temperature to F:

        ``units = {'temperature': 'F'}``
        """

        # define properties in SI units
        # ask if we want unit conversion implemented here?
        self.temperature = temperature
        self.salinity = salinity
        self.sediment = sediment
        self.wave_height = wave_height
        self.fetch = fetch
        self.kinematic_viscosity = 0.000001
        self.name = name

        self.units = self._gnome_units
        if units is not None:
            # self.units is a property, so this is non-destructive
            self.units = units
        super(Water, self).__init__(**kwargs)
        self.Temperature = Temperature(self)
        self.Salinity = Salinity(self)
        self.Sediment = Sediment(self)
        self.WaveHeight = WaveHeight(self)

        # check for obvious errors:
        t = self.get('temperature', 'C')
        if (t > 100) or (t < -10):
            warnings.warn(f"Temperature of {self.temperature} {self._units['temperature']}"
                          "is unrealistic: perhaps you need to specify the units")


    def __repr__(self):
        info = ("{0.__class__.__module__}.{0.__class__.__name__}"
                "(temperature={0.temperature},"
                " salinity={0.salinity})").format(self)
        return info

    __str__ = __repr__

    @property
    def data_start(self):
        '''
        The Water object doesn't directly manage a time series of data,
        so it will not have a data range.
        We will just return an infinite range for Water.
        '''
        return MinusInfTime()

    @property
    def data_stop(self):
        return InfTime()

    def get(self, attr, unit=None):
        '''
        return value in desired unit. If None, then return the value in SI
        units. The user_unit are given in 'units' attribute and each attribute
        carries the value in as given in these user_units.
        '''
        val = getattr(self, attr)

        if val is None:
            return val

        if unit is None:
            # Note: salinity only have one units since we don't
            # have any conversions for them in nucos yet - revisit
            # this per requirements
            if (attr not in self._gnome_units or
                    self._gnome_units[attr] == self._units[attr]):
                return val
            else:
                unit = self._gnome_units[attr]

        if unit == self._units[attr]:  # no need to convert
            return val

        if unit in self._units_type[attr][1]:
            return uc.convert(self._units_type[attr][0], self.units[attr], unit,
                              val)
        else:
            # log to file if we have logger
            ex = uc.InvalidUnitError((unit, self._units_type[attr][0]))
            self.logger.error(str(ex))
            raise ex

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

    # has to be a staticmethod, as the type is not hashable for lru_cache
    @staticmethod
    @lru_cache(2)
    def _get_density(salinity, temp, temp_units):
        '''
        use lru cache so we don't recompute if temp is not changing
        '''
        temp_c = uc.convert(
            'Temperature',
            temp_units,
            # self.units['temperature'],
            'C',
            temp)
        # sea level pressure in decibar - don't expect atmos_pressure to change
        # also expect constants to have SI units
        rho = gsw.rho(salinity, temp_c, constants.atmos_pressure * 0.0001)

        return rho

    @property
    def density(self):
        '''
        return the density based on water salinity and temperature. The
        salinity is in 'psu'; it is not being converted to absolute salinity
        units - for our purposes, this is sufficient. Using gsw.rho()
        internally which expects salinity in absolute units.
        '''
        return self._get_density(self.salinity, self.temperature,
                                 self.units['temperature'])

    @property
    def units(self):
        if not hasattr(self, '_units'):
            self._units = {}

        return self._units

    @units.setter
    def units(self, u_dict):
        if not hasattr(self, '_units'):
            self._units = {}

        for prop, unit in u_dict.items():
            if prop in self._units_type:
                if unit not in self._units_type[prop][1]:
                    msg = ("{0} are invalid units for {1}.  Ignore it.".format(
                        unit, prop))
                    self.logger.error(msg)
                    # should we raise error?
                    raise uc.InvalidUnitError(msg)

            # allow user to add new keys to units dict.
            # also update prop if unit is valid
            self._units[prop] = unit

    def _convert_sediment_units(self, from_, to):
        '''
        used internally to convert to/from sediment units.
        '''
        if from_ == to:
            return self.sediment

        if from_ == 'mg/l':
            # convert to kg/m^3
            return self.sediment / 1000.0
        else:
            return self.sediment * 1000.0

# from gnome.environment.gridded_objects_base import VectorVariable

# class WaterConditions(VectorVariable, Environment):
#     '''
#     An aggregated object describing the water conditions.
#     '''
