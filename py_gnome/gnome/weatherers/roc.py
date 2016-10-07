'''
oil removal from various cleanup options
add these as weatherers
'''
import datetime
import copy
import unit_conversion as uc

from colander import (SchemaNode, MappingSchema, Integer, Float, String, OneOf)

from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable, Field
from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue1dArraySchema)
from gnome.persist import validators, base_schema

from .core import WeathererSchema
from .. import _valid_units


# define valid units at module scope because the Schema and Object both use it
_valid_dist_units = _valid_units('Length')
_valid_vel_units = _valid_units('Velocity')

class UnitsSchema(MappingSchema):
    offset = SchemaNode(String(),
                       description='SI units for distance',
                       validator=OneOf(_valid_dist_units))

    boom_length = SchemaNode(String(),
                            description='SI units for distance',
                            validator=OneOf(_valid_dist_units))

    boom_draft = SchemaNode(String(),
                            description='SI units for distance',
                            validator=OneOf(_valid_dist_units))

    speed = SchemaNode(String(),
                       description='SI units for speed',
                       validator=OneOf(_valid_vel_units))

class OnSceneTupleSchema(DefaultTupleSchema):
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
                          default=base_schema.now,
                          validator=validators.convertible_to_seconds)

class OnSceneTimeSeriesSchema(DatetimeValue1dArraySchema):
    value = OnSceneTupleSchema(default=(datetime.datetime.now(),
                               datetime.datetime.now()))

    def validator(self, node, cstruct):
        '''
        validate on-scene timeseries numpy array
        '''
        validators.no_duplicate_datetime(node, cstruct)
        validators.ascending_datetime(node, cstruct)

class BurnSchema(WeathererSchema):
    offset = SchemaNode(Integer())
    boom_length = SchemaNode(Integer())
    boom_draft = SchemaNode(Integer())
    speed = SchemaNode(Float())
    throughput = SchemaNode(Float())
    burn_effeciency_type = SchemaNode(String())
    units = UnitsSchema()
    timeseries = OnSceneTimeSeriesSchema()

class Burn(Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('offset', save=True, update=True),
               Field('boom_length', save=True, update=True),
               Field('boom_draft', save=True, update=True),
               Field('speed', save=True, update=True),
               Field('timeseries', save=True, update=True),
               Field('throughput', save=True, update=True),
               Field('burn_effeciency_type', save=True, update=True),
               Field('units', save=True, update=True)]

    _schema = BurnSchema

    _si_units = {'offset': 'm',
                 'boom_length': 'm',
                 'boom_draft': 'in',
                 'speed': 'm/s'}

    _units_type = {'offset': ('offset', _valid_dist_units),
                   'boom_length': ('boom_length', _valid_dist_units),
                   'boom_draft': ('boom_draft', _valid_dist_units),
                   'speed': ('speed', _valid_vel_units)}

    def __init__(self,
                 offset,
                 boom_length,
                 boom_draft,
                 speed,
                 throughput,
                 burn_effeciency_type,
                 timeseries=None,
                 units=_si_units,
                 **kwargs):
        
        self.offset = offset
        self.boom_length = boom_length
        self.boom_draft = boom_draft
        self.speed = speed
        self._units = dict(self._si_units)
        self.units = units
        self.timeseries = timeseries

    def get(self, attr, unit=None):
        '''
        return value in desired unit. If None, then return the value in SI
        units. The user_unit are given in 'units' attribute and each attribute
        carries the value in as given in these user_units.
        '''
        val = getattr(self, attr)


    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, u_dict):
        for prop, unit in u_dict.iteritems():
            if prop in self._units_type:
                if unit not in self._units_type[prop][1]:
                    msg = ("{0} are invalid units for {1}."
                           "Ignore it".format(unit, prop))
                    self.logger.error(msg)
                    raise uc.InvalidUnitError(msg)

            self._units[prop] = unit

