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
from gnome.persist.extend_colander import LocalDateTime, DefaultTupleSchema, NumpyArray
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
    start = SchemaNode(LocalDateTime(default_tzinfo=None),
                       validator=validators.convertible_to_seconds)

    stop = SchemaNode(LocalDateTime(default_tzinfo=None),
                       validator=validators.convertible_to_seconds)

class OnSceneTimeSeriesSchema(NumpyArray):
    value = OnSceneTupleSchema()

    def validator(self, node, cstruct):
        '''
        validate on-scene timeseries list 
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

class Burn(Weatherer, Serializable):
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
                 burn_effeciency_type=1,
                 timeseries=None,
                 units=_si_units,
                 **kwargs):

        super(Burn, self).__init__(**kwargs)

        self.offset = offset
        self.boom_length = boom_length
        self.boom_draft = boom_draft
        self.speed = speed
        self.throughput = throughput
        self.timeseries = timeseries
        self.burn_effeciency_type = burn_effeciency_type
        self._units = dict(self._si_units)
        self.units = units
        self._swath_width = None
        self._area = None
        self._boom_capacity = None
        self._offset_time = None
        self._report = []

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
    
    def get(self, attr, unit=None):
        val = getattr(self, attr)
        if unit is None:
            if (attr not in self._si_units or
                    self._is_units[attr] == self.units[attr]):
                return val
            else:
                unit = self._si_units[attr]

        if unit in self._units_type[attr][1]:
            return uc.convert(self._units_type[attr][0], self.units[attr],
                             unit, val)
        else:
            ex = uc.InvalidUnitError((unit, self._units_type[attr][0]))
            self.logger.error(str(ex))
            raise ex

    def set(self, attr, value, unit):
        if unit not in self._units_type[attr][0]:
            raise uc.InvalidUnitError((unit, self._units_type[attr][0]))
        
        setattr(self, attr, value)
        self.units[attr] = unit

    def prepare_for_model_run(self, sc):
        self._setup_report(sc)
        self._swath_width = 0.3 * self.boom_length
        self._area = self._swath_width * (0.4125 * self.boom_length / 3) * 2/3
        self._boom_capacity = self.boom_draft / 36 * self._area
        self._offset_time = self.offset * 0.00987 / self.speed
        self._area_coverage_rate = self._swath_width * self.speed / 430

        if self._swath_width > 1000:
            self.report.append('Swaths > 1000 feet may not be achievable in the field')

        if self.on:
            sc.mass_balance['burned'] = 0.0
            sc.mass_balance[self.id] = 0.0

        _is_burning = False
        _is_retired = False

        _time_collecting_in_sim = 0.
        _total_burns = 0.
        _time_burning = 0.

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        1. set 'active' flag based on timeseries and model_time
        2. Mark LEs to be burned, do them in order right now. assume all LEs
           that are released together will be burned together since they would
           be closer to each other in position.
        '''
        if not self.active:
            return
        
        if self._is_active(model_time, time_step):
            self._active = True
        else:
            self._active = False
        
        


    def weather_elements(self, sc, time_step, mdoel_time):

        if not self.active or len(sc) == 0:
            return



    def _is_active(self, model_time, time_step):
        for t in self.timeseries:
            print model_time
            if model_time >= t[0] and model_time + datetime.timedelta(seconds=time_step/2) <= t[1]:
                return True

        return False
    
    def _setup_report(self, sc):
        if 'report' not in sc:
            sc.report = {}

        sc.report[self.id] = []  
        self.report = sc.report[self.id]

   
