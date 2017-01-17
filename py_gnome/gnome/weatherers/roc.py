'''
oil removal from various cleanup options
add these as weatherers
'''
from __future__ import division

import datetime
import copy
import unit_conversion as uc
import json

from colander import (drop, SchemaNode, MappingSchema, Integer, Float, String, OneOf)

from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable, Field
from gnome.persist.extend_colander import LocalDateTime, DefaultTupleSchema, NumpyArray
from gnome.persist import validators, base_schema

from .core import WeathererSchema
from .. import _valid_units


# define valid units at module scope because the Schema and Object both use it
_valid_dist_units = _valid_units('Length')
_valid_vel_units = _valid_units('Velocity')
_valid_vol_units = _valid_units('Volume')
_valid_dis_units = _valid_units('Discharge')

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

class ResponseSchema(WeathererSchema):
    timeseries = OnSceneTimeSeriesSchema()

class Response(Weatherer, Serializable):

    def __init__(self, **kwargs):
        super(Response, self).__init__(**kwargs)
        self._report = []

    def _get_thickness(self, sc):
        oil_thickness = 0.0
        substance = self._get_substance(sc)
        if sc['area'].any() > 0:
            volume_emul = (sc['mass'].mean() / substance.get_density()) / (1.0 - sc['frac_water'].mean())
            oil_thickness = volume_emul / sc['area'].mean()

        return uc.convert('Length', 'meters', 'inches', oil_thickness)

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

    def _is_active(self, model_time, time_step):
        for t in self.timeseries:
            if model_time >= t[0] and model_time + datetime.timedelta(seconds=time_step / 2) <= t[1]:
                return True

        return False

    def _setup_report(self, sc):
        if 'report' not in sc:
            sc.report = {}

        sc.report[self.id] = []
        self.report = sc.report[self.id]

    def _get_substance(self, sc):
        '''
        return a single substance - recovery options only know about the
        total amount removed. Unclear how to assign this to multiple substances
        for now, just log an error if more than one substance is present
        '''
        substance = sc.get_substances(complete=False)
        if len(substance) > 1:
            self.logger.error('Found more than one type of oil '
                              '- not supported. Results with be incorrect')

        return substance[0]

class Platform(Serializable):

    with open('platforms.json', 'r') as f:
        _types = json.load(f)
        _types['vessel'] = dict([t['name'] for t in _types['vessel']], _types['vessel'])
        _types['aircraft'] = dict([t['name'] for t in _types['aircraft']], _types['aircraft'])

    def __init__(self,
                 type_,
                 kind=None,
                 **kwargs):
        if kwargs is None:
            if kind is None:
                n = 'Typical Large Vessel' if type_ == 'vessel' else 'Test Platform'
                for k, attr in Platform._types[type_][n]:
                    setattr(self, k, attr)
        else:
            if kind is not None:
                for k, attr in Platform._types[type_][kind]:
                    setattr(self, k, attr)
            for k, attr in kwargs:
                setattr(self, k, attr)
        self.type = type_
        self._state = None
        self._time_to_next = None

    def release_rate(self, dosage):
        return (dosage * self.application_speed * self.swadth_width) / 430

    def one_way_transit_time(self, dist):
        raw = dist / self.transit_speed
        if self.type == 'aircraft':
            raw += self.taxi_land_depart / 60
        return raw * 60

    def adjusted_op_time(self, dist, op_time):
        pass


    def max_dosage(self):
        return (self.pump_rate_max * 430) / (self.application_speed_min * self.swath_width_min)

    def min_dosage(self):
        return (self.pump_rate_min * 430) / (self.application_speed_max * self.swath_width_max)


class DisperseUnitsSchema(MappingSchema):
    pass

class DisperseSchema(ResponseSchema):
    pass

class Disperse(Response):



    _state = copy.deepcopy(Response._state)
    _state += [Field('offset', save=True, update=True)]

    def __init__(self,
                 name=None,
                 transit=None,
                 transit_unit='nm',
                 pass_length=4,
                 pass_length_unit='nm',
                 dosage=None,
                 dosage_unit='Gallon/Acre',
                 cascade=None,
                 platform=None,
                 platform_type='',
                 edac='0',
                 loading_type='simultaneous',
                 pass_type='bidirectional',
                 **kwargs):
        super(Disperse, self).__init__(**kwargs)
        self.name = name
        self.transit = transit
        self.transit_unit = transit_unit
        self.pass_length = pass_length
        self.pass_length_unit = pass_length_unit
        self.dosage = dosage
        self.dosage_unit = dosage_unit
        self.cascade = cascade
        self.platform = platform
        self._platform = None
        self.platform_type = platform_type
        self.edac = edac
        self.loading_type = loading_type
        self.pass_type = pass_type

    @property
    def platform(self):
        return self._platform.to_dict()

    @platform.setter()
    def platform(self, plt):
        _type = None
        if 'type' in plt.keys():
            _type = plt.pop('type')
        elif plt['name'] in Platform._types['aircraft'].keys():
            _type = 'aircraft'
        elif plt['name'] in Platform._types['vessel'].keys():
            _type = 'vessel'
        else:
            raise ValueError('Must specify platform type (aircraft/vessel)')
        self._platform = Platform(_type,
                                  kind=plt.pop('name', None),
                                  **plt)

    def prepare_for_model_run(self, sc):
        self._setup_report(sc)
        self._swath_width = 0.3 * self.boom_length
        self._area = self._swath_width * (0.4125 * self.boom_length / 3) * 2 / 3
        self._boom_capacity = self.boom_draft / 36 * self._area
        self._boom_capacity_remaining = self._boom_capacity
        self._offset_time = (self.offset * 0.00987 / self.speed) * 60
        self._area_coverage_rate = self._swath_width * self.speed / 430

        if self._swath_width > 1000:
            self.report.append('Swaths > 1000 feet may not be achievable in the field')

        if self.speed > 1.2:
            self.report.append('Excessive entrainment of oil likely to occur at speeds greater than 1.2 knots.')

        if self.on:
            sc.mass_balance['burned'] = 0.0
            sc.mass_balance[self.id] = 0.0
            sc.mass_balance['boomed'] = 0.0

        self._is_collecting = True


class BurnUnitsSchema(MappingSchema):
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

class BurnSchema(ResponseSchema):
    offset = SchemaNode(Integer())
    boom_length = SchemaNode(Integer())
    boom_draft = SchemaNode(Integer())
    speed = SchemaNode(Float())
    throughput = SchemaNode(Float())
    burn_efficiency_type = SchemaNode(String())
    units = BurnUnitsSchema()

class Burn(Response):
    _state = copy.deepcopy(Response._state)
    _state += [Field('offset', save=True, update=True),
               Field('boom_length', save=True, update=True),
               Field('boom_draft', save=True, update=True),
               Field('speed', save=True, update=True),
               Field('timeseries', save=True, update=True),
               Field('throughput', save=True, update=True),
               Field('burn_efficiency_type', save=True, update=True),
               Field('units', save=True, update=True)]

    _schema = BurnSchema

    _si_units = {'offset': 'ft',
                 'boom_length': 'ft',
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
                 burn_efficiency_type=1,
                 timeseries=None,
                 units=_si_units,
                 **kwargs):

        super(Burn, self).__init__(**kwargs)

        self.offset = offset
        self._units = dict(self._si_units)
        self.units = units
        self.boom_length = boom_length
        self.boom_draft = boom_draft
        self.speed = speed
        self.throughput = throughput
        self.timeseries = timeseries
        self.burn_efficiency_type = burn_efficiency_type
        self._swath_width = None
        self._area = None
        self._boom_capacity = None
        self._offset_time = None

        self._is_collecting = False
        self._is_burning = False
        self._is_boom_filled = False
        self._is_transiting = False
        self._is_cleaning = False

        self._time_collecting_in_sim = 0.
        self._total_burns = 0.
        self._time_burning = 0.
        self._ts_burned = 0.
        self._ts_collected = 0.
        self._burn_time = None

    def prepare_for_model_run(self, sc):
        self._setup_report(sc)
        self._swath_width = 0.3 * self.boom_length
        self._area = self._swath_width * (0.4125 * self.boom_length / 3) * 2 / 3
        self._boom_capacity = self.boom_draft / 36 * self._area
        self._boom_capacity_remaining = self._boom_capacity
        self._offset_time = (self.offset * 0.00987 / self.speed) * 60
        self._area_coverage_rate = self._swath_width * self.speed / 430

        if self._swath_width > 1000:
            self.report.append('Swaths > 1000 feet may not be achievable in the field')

        if self.speed > 1.2:
            self.report.append('Excessive entrainment of oil likely to occur at speeds greater than 1.2 knots.')

        if self.on:
            sc.mass_balance['burned'] = 0.0
            sc.mass_balance[self.id] = 0.0
            sc.mass_balance['boomed'] = 0.0

        self._is_collecting = True

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        1. set 'active' flag based on timeseries and model_time
        2. Mark LEs to be burned, do them in order right now. assume all LEs
           that are released together will be burned together since they would
           be closer to each other in position.
        '''

        self._ts_collected = 0.
        self._ts_burned = 0.

        if self._is_active(model_time, time_step):
            self._active = True
        else:
            self._active = False

        if not self.active:
            return

        self._time_remaining = time_step

        while self._time_remaining > 0.:
            if self._is_collecting:
                self._collect(sc, time_step, model_time)

            if self._is_transiting and self._is_boom_full:
                self._transit(sc, time_step, model_time)

            if self._is_burning:
                self._burn(sc, time_step, model_time)

            if self._is_cleaning:
                self._clean(sc, time_step, model_time)

            if self._is_transiting and not self._is_boom_full:
                self._transit(sc, time_step, model_time)

    def _collect(self, sc, time_step, model_time):
        # calculate amount collected this time_step
        if self._burn_time is None:
            self._burn_rate = 0.14 * (100 - (sc['frac_water'].mean() * 100)) / 100
            self._burn_time = (0.33 * self.boom_draft / self._burn_rate) * 60
            self._burn_time_remaining = self._burn_time

        oil_thickness = self._get_thickness(sc)
        encounter_rate = 63.13 * self._swath_width * oil_thickness * self.speed
        emulsion_rr = encounter_rate * self.throughput
        if oil_thickness > 0:
            # old ROC equation
            # time_to_fill = (self._boom_capacity_remaining / emulsion_rr) * 60
            # new ebsp equation
            time_to_fill = ((self._boom_capacity_remaining * 0.17811) * 42) / emulsion_rr
        else:
            time_to_fill = 0.

        if time_to_fill > self._time_remaining:
            # doesn't finish fill the boom in this time step
            self._ts_collected = emulsion_rr * (self._time_remaining / 60)
            self._boom_capacity_remaining -= collected
            self._time_remaining = 0.0
            self._time_collecting_in_sim += self._time_remaining
        elif self._time_remaining > 0:
            # finishes filling the boom in this time step any time remaining
            # should be spend transiting to the burn position
            self._ts_collected = self._boom_capacity_remaining

            self._boom_capacity_remaining = 0.0
            self._is_boom_full = True

            self._time_remaining -= time_to_fill
            self._time_collecting_in_sim += time_to_fill
            self._offset_time_remaining = self._offset_time
            self._is_collecting = False
            self._is_transiting = True

    def _transit(self, sc, time_step, model_time):
        # transiting to burn site
        # does it arrive and start burning?
        if self._time_remaining > self._offset_time_remaining:
            self._time_remaining -= self._offset_time_remaining
            self._offset_time_remaining = 0.
            self._is_transiting = False
            if self._is_boom_full:
                self._is_burning = True
            else:
                self._is_collecting = True
        elif self._time_remaining > 0:
            self._offset_time_remaining -= self._time_remaining
            self._time_remaining = 0.

    def _burn(self, sc, time_step, model_time):
        # burning
        self._is_boom_full = False
        if self._time_remaining > self._burn_time_remaining:
            self._time_remaining -= self._burn_time_remaining
            self._burn_time_remaining = 0.
            burned = self._boom_capacity - self._boom_capacity_remaining
            self._ts_burned = burned
            self._is_burning = False
            self._is_cleaning = True
            self._cleaning_time_remaining = 3600  # 1hr in seconds
        elif self._time_remaining > 0:
            frac_burned = self._time_remaining / self._burn_time
            burned = self._boom_capacity * frac_burned
            self._boom_capacity_remaining += burned
            self._ts_burned = burned
            self._burn_time_remaining -= self._time_remaining
            self._time_remaining = 0.

    def _clean(self, sc, time_step, model_time):
        # cleaning
        self._burn_time = None
        if self._time_remaining > self._cleaning_time_remaining:
            self._time_remaining -= self._cleaning_time_remaining
            self._cleaning_time_remaining = 0.
            self._is_cleaning = False
            self._is_transiting = True
            self._offset_time_remaining = self._offset_time
        elif self._time_remaining > 0:
            self._cleaning_time_remaining -= self._time_remaining
            self._time_remaining = 0.

    def weather_elements(self, sc, time_step, model_time):
        '''
        Remove mass from each le equally for now, no flagging for not
        just make sure it's from floating oil.
        '''
        if not self.active or len(sc) == 0:
            return

        les = sc.itersubstancedata(self.array_types)
        for substance, data in les:
            if len(data['mass']) is 0:
                continue

            if self._ts_collected:
                sc.mass_balance['boomed'] += self._ts_collected
                sc.mass_balance[self.id] += self._ts_collected
                total_mass = data['mass'].sum()
                rm_mass_frac = min(self._ts_collected / total_mass, 1.0)
                data['mass_components'] = \
                    (1 - rm_mass_frac) * data['mass_components']
                data['mass'] = data['mass_components'].sum(1)

                self.logger.debug('{0} amount boomed for {1}: {2}'
                                  .format(self._pid, substance.name, self._ts_collected))

            if self._ts_burned:
                sc.mass_balance['burned'] += self._ts_burned
                sc.mass_balance['boomed'] -= self._ts_burned



