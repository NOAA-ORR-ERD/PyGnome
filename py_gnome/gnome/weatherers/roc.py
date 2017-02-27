'''
oil removal from various cleanup options
add these as weatherers
'''
from __future__ import division

import pytest
import datetime
import copy
import unit_conversion as uc
import json
import os
import logging
import numpy as np
import math
from collections import OrderedDict

from colander import (drop, SchemaNode, MappingSchema, Integer, Float, String, OneOf, Mapping)

from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable, Field
from gnome.persist.extend_colander import LocalDateTime, DefaultTupleSchema, NumpyArray, TimeDelta
from gnome.persist import validators, base_schema

from gnome.weatherers.core import WeathererSchema
from gnome import _valid_units
from gnome.basic_types import oil_status, fate as bt_fate


# define valid units at module scope because the Schema and Object both use it
_valid_dist_units = _valid_units('Length')
_valid_vel_units = _valid_units('Velocity')
_valid_vol_units = _valid_units('Volume')
_valid_dis_units = _valid_units('Discharge')
_valid_time_units = _valid_units('Time')
_valid_oil_concentration_units = _valid_units('Oil Concentration')
_valid_concentration_units = _valid_units('Concentration In Water')


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
            volume_emul = (sc['mass'].mean() / substance.density_at_temp()) / (1.0 - sc['frac_water'].mean())
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
                    self._si_units[attr] == self.units[attr]):
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

    def _remove_mass_simple(self, data, amount):
        total_mass = data['mass'].sum()
        rm_mass_frac = min(amount / total_mass, 1.0)
        data['mass_components'] = \
            (1 - rm_mass_frac) * data['mass_components']
        data['mass'] = data['mass_components'].sum(1)

    def _remove_mass_indices(self, data, amounts, indices):
        #removes mass from the mass components specified by an indices array
        masses = data['mass'][indices]
        rm_mass_frac = np.clip(amounts / masses, 0, 1)
        data['mass_components'][indices] = (1 - rm_mass_frac)[:, np.newaxis] * data['mass_components'][indices]
        data['mass'][indices] = data['mass_components'][indices].sum(1)

    def index_of(self, time):
        '''
        Returns the index of the timeseries entry that the time specified is within.
        If it is not in one of the intervals, -1 will be returned
        '''
        for i, t in enumerate(self.timeseries):
            if time >= t[0] and time < t[-1]:
                return i
        return -1

    def next_interval_index(self, time):
        '''
        returns the index of the next interval, even if outside interval.
        returns None if there is no next interval
        '''
        if time >= self.timeseries[-1][-1]:
            #off end
            return None
        if time < self.timeseries[0][0]:
            #before start
            return 0
        idx = self.index_of(time)
        if idx > -1:
            #inside valid interval
            return idx + 1 if idx + 1 != len(self.timeseries) else None
        if idx == -1:
            #outside timeseries intervals
            for i, t in enumerate(self.timeseries[0:-1]):
                if time >= self.timeseries[i][-1] and time < self.timeseries[i+1][0]:
                    return i+1

    def time_to_next_interval(self, time):
        '''
        if within an interval, returns time left in the interval.
        if between intervals, returns time until start of next interval
        if past end, or response deactivated, return None
        '''
        cur_idx = self.index_of(time)
        if cur_idx == -1:
            next_idx = self.next_interval_index(time)
            if next_idx is None:
                return None
            else:
                return self.timeseries[next_idx][0] - time
        else:
            return self.timeseries[cur_idx][-1] - time

    def is_operating(self, time):
        return self.index_of(time) > -1


class PlatformUnitsSchema(MappingSchema):
    def __init__(self, *args, **kwargs):
        for k, v in Platform._attr.items():
            self.add(SchemaNode(String(), missing=drop, name=k, validator=OneOf(v[2])))
        super(PlatformUnitsSchema, self).__init__()


class PlatformSchema(base_schema.ObjType):

    def __init__(self, *args, **kwargs):
        for k in Platform._attr.keys():
            self.add(SchemaNode(Float(), missing=drop, name=k))
        units = PlatformUnitsSchema()
        units.missing = drop
        units.name = 'units'
        self.add(units)
        super(PlatformSchema, self).__init__()


class Platform(Serializable):

    _attr = {"swath_width_max": ('ft', 'length', _valid_dist_units),
             "swath_width": ('ft', 'length', _valid_dist_units),
             "swath_width_min": ('ft', 'length', _valid_dist_units),
             "reposition_speed": ('kts', 'velocity', _valid_vel_units),
             "application_speed_min": ('kts', 'velocity', _valid_vel_units),
             "application_speed": ('kts', 'velocity', _valid_vel_units),
             "application_speed_max": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_max_without_payload": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_without_payload": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_min_without_payload": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_with_payload": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_max_with_payload": ('kts', 'velocity', _valid_vel_units),
             "cascade_transit_speed_min_with_payload": ('kts', 'velocity', _valid_vel_units),
             "transit_speed_max": ('kts', 'velocity', _valid_vel_units),
             "transit_speed_min": ('kts', 'velocity', _valid_vel_units),
             "transit_speed": ('kts', 'velocity', _valid_vel_units),
             "fuel_load": ('min', 'time', _valid_time_units),
             "taxi_time_landing": ('min', 'time', _valid_time_units),
             "staging_area_brief": ('min', 'time', _valid_time_units),
             "dispersant_load": ('min', 'time', _valid_time_units),
             "taxi_land_depart": ('min', 'time', _valid_time_units),
             "taxi_time_takeoff": ('min', 'time', _valid_time_units),
             "u_turn_time": ('min', 'time', _valid_time_units),
             "max_op_time": ('hr', 'time', _valid_time_units),
             "max_range_no_payload": ('nm', 'length', _valid_dist_units),
             "max_range_with_payload": ('nm', 'length', _valid_dist_units),
             "approach": ('nm', 'length', _valid_dist_units),
             "departure": ('nm', 'length', _valid_dist_units),
             "payload": ('gal', 'volume', _valid_vol_units),
             "pump_rate_max": ('gal/min', 'discharge', _valid_dis_units),
             "pump_rate_min": ('gal/min', 'discharge', _valid_dis_units)}

    _si_units = dict([(k, v[0]) for k, v in _attr.items()])

    _units_type = dict([(k, (v[1], v[2])) for k, v in _attr.items()])

    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, 'platforms.json'), 'r') as f:
        js = json.load(f)
        plat_types = dict(zip([t['name'] for t in js['vessel']], js['vessel']))
        plat_types.update(dict(zip([t['name'] for t in js['aircraft']], js['aircraft'])))

    _schema = PlatformSchema

    _state = copy.deepcopy(Serializable._state)

    _state += [Field(k, save=True, update=True) for k in _attr.keys()]
    _state += [Field('units', save=True, update=True)]

    def __init__(self,
                 units=None,
                 **kwargs):

        if '_name' in kwargs.keys():
            kwargs = self.plat_types[kwargs.pop('_name')]
        if units is None:
            units = dict([(k, v[0]) for k, v in self._attr.items()])
        self.units = units
        for k in Platform._attr.keys():
            setattr(self, k, kwargs.get(k, None))

        self.disp_remaining = 0
        self.cur_pump_rate = 0

        super(Platform, self).__init__()

    def get(self, attr, unit=None):
        val = getattr(self, attr)
        if unit is None:
            if (attr not in self._si_units or
                    self._si_units[attr] == self.units[attr]):
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

    def release_rate(self, dosage, unit='gal/acre'):
        '''return unit = gal/min'''
        if unit != 'gal/acre':
            dosage = uc.Convert('oilconcentration', 'unit', 'gal/acre', dosage)
        a_s = self.get('application_speed', 'ft/min')
        s_w = self.get('swadth_width', 'ft')

        return uc.convert('area', 'ft^2', 'acre', (dosage * a_s * s_w))

    @classmethod
    def new_from_dict(cls, dict_):
        '''
        Need to override this, because what the default one does is insane
        '''
        return cls(**dict_)

    def one_way_transit_time(self, dist, unit='nm', payload=False):
        '''return unit = sec'''
        t_s = self.get('transit_speed', 'kts')
        t_l_d = self.get('taxi_land_depart', 'sec')
        raw = dist / t_s * 3600
        if t_l_d is not None:
            raw += t_l_d
        return raw

    def max_dosage(self):
        '''return unit = gal/acre'''
        p_r_m = self.get('pump_rate_max', 'm^3/s')
        a_s = self.get('application_speed', 'm/s')
        s_w_m = self.get('swath_width_min', 'm')
        dos = (p_r_m) / (a_s * s_w_m)
        dos = uc.convert('length', 'm', 'micron', dos)
        dos = uc.convert('oilconcentration', 'micron', 'gal/acre', dos)
        return dos

    def min_dosage(self):
        '''return unit = gal/acre'''
        p_r_m = self.get('pump_rate_min', 'm^3/s')
        a_s = self.get('application_speed', 'm/s')
        s_w_m = self.get('swath_width_max', 'm')
        dos = (p_r_m) / (a_s * s_w_m)
        dos = uc.convert('length', 'm', 'micron', dos)
        dos = uc.convert('oilconcentration', 'micron', 'gal/acre', dos)
        return dos

    def cascade_time(self, dist, unit='nm', payload=False):
        '''return unit = sec'''
        dist = dist if unit == 'nm' else uc.convert('length', unit, 'nm', dist)
        max_range = self.get('max_rage_with_payload', 'nm') if payload else self.get('max_range_no_payload', 'nm')
        speed = self.get('cascade_transit_speed_with_payload', 'kts') if payload else self.get('cascade_transit_speed_without_payload', 'kts')
        taxi_land_depart = self.get('taxi_land_depart', 'hr')
        fuel_load = self.get('refuel', 'hr')

        cascade_time = 0
        if dist > max_range:
            num_legs = dist / max_range
            frac_leg = (num_legs * 1000) % 1000
            num_legs = int(num_legs)
            cascade_time += taxi_land_depart
            cascade_time += (num_legs * max_range)
            inter_stop = (taxi_land_depart * 2 + fuel_load)
            cascade_time += num_legs * inter_stop
            cascade_time += frac_leg * (max_range / speed)
            cascade_time += taxi_land_depart
        else:
            cascade_time += taxi_land_depart * 2
            cascade_time += dist / speed
        return cascade_time * 3600

    def max_onsite_time(self, dist, simul=False):
        '''
        return time in sec
        '''
        m_o_t = self.get('max_op_time', 'sec')
        o_w_t_t = self.one_way_transit_time(dist)
        r_r = self.refuel_reload(simul=simul)
        rv = m_o_t - o_w_t_t * 2 - r_r
#         if rv < 0:
#             logging.warn('max onsite time is less than zero')
#         else:
#             pld = self.get('payload', 'gal')
#             m_p_r = self.get('max_pump_rate', 'gal/hr')
#             if rv < (pld / m_p_r):
#                 logging.warn("max onsite time is less than possible time to finsish spraying")
        return rv

    def num_passes_possible(self, time, pass_len):
        '''
        In a given time (sec) compute maximum number of complete passes before
        needing to return to base.

        A pass consists of an approach, spray, u-turn, and reposition.
        '''

#         rep = self.get('reposition_speed', 'm/s')

        return int(time.total_seconds() / int(self.pass_duration(pass_len)))

    def refuel_reload(self, simul=False):
        '''return unit = sec'''
        rl = self.get('dispersant_load', 'sec')
        rf = self.get('fuel_load', 'sec')
        return max(rl, rf) if simul else rf + rl

    def pass_duration(self, pass_len, units='nm', bidirectional=False):
        '''
        pass_len in nm
        return in sec
        '''
        appr_dist = self.get('approach', 'm')
        dep_dist = self.get('departure', 'm')
        rep_speed = self.get('reposition_speed', 'm/s')
        appr_time = appr_dist / rep_speed
        dep_time = dep_dist / rep_speed
        u_turn = self.get('u_turn_time', 'sec')

        pass_len = uc.convert('length', units, 'm', pass_len)
        app_speed = self.get('application_speed', 'm/s')
        spray_time = pass_len / app_speed
        if bidirectional == True:
            return appr_time + spray_time + u_turn + spray_time + dep_time
        else:
            return appr_time + spray_time + u_turn + appr_time + dep_time

    def sortie_possible(self, time_avail, transit, pass_len):
        # assume already refueled/reloaded
        # possible if able to complete transit, at least one pass, and transit back within time available
        min_spray_time = self.pass_duration(pass_len)
        tot_mission_time = self.one_way_transit_time(transit) * 2 + min_spray_time
        return time_avail > datetime.timedelta(seconds=tot_mission_time)

    def eff_pump_rate(self, dosage, unit='gal/acre'):
        '''
        given a dosage, determine the pump rate necessary given the airspeed and area covered in a pass
        return value = m^3/s
        '''
        dosage = uc.convert('oilconcentration', unit, 'micron', dosage)
        dosage = uc.convert('length', 'micron', 'm', dosage)
        app_speed = self.get('application_speed', 'm/s')
        swath_width = self.get('swath_width', 'm')
        eff_pr = dosage * app_speed * swath_width
        max_pr = self.get('pump_rate_max', 'm^3/s')
        if eff_pr > max_pr:
            #log warning?
            print 'computed pump rate is too high for this platform. using max instead'
            return max_pr
        else:
            return eff_pr

class DisperseUnitsSchema(MappingSchema):
    def __init__(self, *args, **kwargs):
        for k, v in Disperse._attr.items():
            self.add(SchemaNode(String(), missing=drop, name=k, validator=OneOf(v[2])))
        super(DisperseUnitsSchema, self).__init__()


class DisperseSchema(base_schema.ObjType):
    loading_type = SchemaNode(String(), validator=OneOf(['simultaneous', 'separate']))
    dosage_type = SchemaNode(String(), missing=drop, validator=OneOf(['auto', 'custom']))
    disp_oil_ratio = SchemaNode(Float(), missing=drop)
    disp_eff = SchemaNode(Float(), missing=drop)
    platform = PlatformSchema()
    timeseries = OnSceneTimeSeriesSchema()

    def __init__(self, *args, **kwargs):
        for k, v in Disperse._attr.items():
            self.add(SchemaNode(Float(), missing=drop, name=k))
        units = DisperseUnitsSchema()
        units.missing = drop
        units.name = 'units'
        self.add(units)
        super(DisperseSchema, self).__init__()

class Disperse(Response):

    _attr = {'transit': ('nm', 'length', _valid_dist_units),
             'pass_length': ('nm', 'length', _valid_dist_units),
             'cascade_distance': ('nm', 'length', _valid_dist_units),
             'dosage': ('gal/acre', 'oilconcentration', _valid_oil_concentration_units)}

    _si_units = dict([(k, v[0]) for k, v in _attr.items()])

    _units_type = dict([(k, (v[1], v[2])) for k, v in _attr.items()])

    _schema = DisperseSchema

    _state = copy.deepcopy(Response._state)

#     _state += [Field(k, save=True, update=True) for k in _attr.keys()]
    _state += [Field('units', save=True, update=True),
               Field('disp_oil_ratio', save=True, update=True),
               Field('disp_eff', save=True, update=True),
               Field('platform', save=True, update=True),
               Field('dosage_type', save=True, update=True),
               Field('loading_type', save=True, update=True),
               Field('timeseries', save=True, update=True),
               Field('report', save=False, update=False),
               Field('wind', save=True, update=True, save_reference=True)]

    wind_eff_list = [15, 30, 45, 60, 70, 78, 80, 82, 83, 84, 84, 84, 84, 84, 83, 83, 82, 80, 79, 78, 77, 75, 73, 71, 69, 67, 65, 63, 60, 58, 55, 53, 50, 47, 44, 41, 38]
    visc_eff_table = OrderedDict([(1, 68), (2, 71), (3, 72.5), (4, 74), (5, 75), (7, 77), (10, 78), (20, 80), (40, 83.5), (70, 85.5), (100, 87), (300, 89.5), (500, 90.5), (700, 91), (1000, 92), (2000, 91), (3000, 83), (5000, 52), (7000, 32), (10000, 17), (20000, 11), (30000, 8.5), (40000, 7), (50000, 6.5), (100000, 6), (1000000, 0)])

    def __init__(self,
                 name=None,
                 transit=None,
                 pass_length=4,
                 dosage=1,
                 dosage_type=None,
                 cascade_on=False,
                 cascade_distance=None,
                 loading_type='simultaneous',
                 pass_type='bidirectional',
                 disp_oil_ratio=None,
                 disp_eff=None,
                 platform=None,
                 units=None,
                 timeseries=None,
                 wind=None,
                 **kwargs):
        super(Disperse, self).__init__(**kwargs)
        self.name = name
        self.transit = transit
        self.pass_length = pass_length
        self.dosage = dosage
        self.dosage_type = dosage_type
        self.cascade_on = cascade_on
        self.cascade_distance = cascade_distance
        self.loading_type = loading_type
        self.pass_type = pass_type
        self.disp_oil_ratio = 20 if disp_oil_ratio is None else disp_oil_ratio
        self.disp_eff = disp_eff
        if self.disp_eff is not None:
            self._disp_eff_type = 'fixed'
        else:
            self._disp_eff_type = 'auto'
        # time to next state
        if platform is not None:
            if isinstance(platform, basestring):
                #find platform name
                self.platform = Platform(_name=platform)
            else:
                #platform is defined as a dict
                self.platform = Platform(**platform)
        else:
            self.platform = platform
        if units is None:
            units = dict([(k, v[0]) for k, v in self._attr.items()])
        self._units = units
        self.timeseries = timeseries
        self.wind = wind
        self.cur_state = None
        self.oil_treated_this_timestep = 0
        self._next_state_time = None
        self._op_start = None
        self._op_end = None
        self._cur_sortie_num = 1
        self._cur_pass_num = 1
        self._area_this_ts = 0
        self._area_this_sortie = 0
        self._disp_sprayed_this_timestep = 0
        self._remaining_dispersant = None

        self._dosage_m = uc.convert('oilconcentration', self.units['dosage'], 'micron', self.dosage)
        self._dosage_m = uc.convert('length', 'micron', 'meters', self._dosage_m)
        self.report=[]
        self.array_types.update({'area', 'density', 'viscosity'})


#     @property
#     def next_state(self):
#         if self.cur_state is None:
#             return None
#         if self.cur_state == 'cascade' or self.cur_state == 'rtb':
#             return 'replenish'
#         elif self.cur_state == 'replenish':
#             return 'en_route'
#         elif self.cur_state == 'en_route':
#             return 'on_site'
#         elif self.cur_state == 'inactive':
#             return 'replenish'
#
#     @property
#     def cur_state_duration(self):
#         if self.cur_state is None:
#             raise ValueError('Current state of None has no duration')
#         if self.cur_state == 'inactive':
#             raise ValueError('inactive has special duration and should not be requested')
#         if self.cur_state == 'cascade':
#             return self.platform.cascade_time(self.cascade_distance)
#         if self.cur_state == 'ready':
#             return self.platform.refuel_reload(self.loading_type)
#         if self.cur_state == 'en_route':
#             return self.platform.one_way_transit_time(self.transit)
#         if self.cur_state == 'on_site':
#             return self.platform.max_onsite_time(self.transit)
#         if self.cur_state == 'returning':
#             return self.platform.one_way_transit_time(self.transit)

    def get_mission_data(self,
                         dosage=None,
                         area=None,
                         pass_len=None,
                         efficiency=None,
                         units=None):
        '''
        Given a dosage and an area to spray, will return a tuple of information as follows:
        Minimize number of passes by using high swath_width. If pump rate cannot get to the dosage necessary
        reduce the swath width until it can.
        Default units are ('gal/acre', 'm^3, 'nm', percent)
        Return tuple is as below
        (num_passes, disp/pass, oil/pass)
        (number, gal, ft, gal/min)
        '''
        if units is None:
            units = {'dosage': 'gal/acre',
                     'area': 'm^3',
                     'pass_len': 'nm',
                     'efficiency': 'percent'}

        # Efficiency determines how much of the pass length is
        pass_area = self.get('swath_width', 'm') * uc.convert('length', units['pass_len'], 'm', pass_len)
        pass_len = uc.convert('length', units['pass_len'], 'm', pass_len)
        app_speed = self.get('application_speed', 'm/s')
        spray_time = pass_len / app_speed
        max_dos = (self.get('pump_rate_max', 'm^3/s') * spray_time / pass_area)
        max_dos = uc.convert('length', 'm', 'micron', max_dos)
        max_dos = uc.convert('oilconcentration', 'micron', 'gal/acre', max_dos)

    def prepare_for_model_run(self, sc):
        self._setup_report(sc)
        if self.on:
            sc.mass_balance[self.id] = 0.0
            sc.mass_balance['dispersed'] = 0.0
        if self.cascade_on:
            self.cur_state = 'cascade'
        else:
            self.cur_state = 'retired'
        self._remaining_dispersant = self.platform.get('payload', 'm^3')
        self.oil_treated_this_timestep = 0

    def get_disp_eff(self, sc, model_time):
        wind_eff_list = Disperse.wind_eff_list
        visc_eff_table = Disperse.visc_eff_table
        vel = self.wind.get_value(model_time)
        spd = math.sqrt(vel[0]**2 + vel[1]**2)
        wind_eff = wind_eff_list[int(spd)] / 100.
        idxs = np.where(sc['viscosity'] * 1000000 < 5000)[0]
        avg_visc = np.mean(sc['viscosity'][idxs] * 1000000) if len(idxs) > 0 else 1000000
        print 'visc = ', avg_visc
        visc_eff = visc_eff_table[visc_eff_table.keys()[np.searchsorted(visc_eff_table.keys(), avg_visc)]] / 100
        return wind_eff * visc_eff

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        '''

        if self._disp_eff_type != 'fixed':
            self.disp_eff = self.get_disp_eff(sc, model_time)
#             print 'efficiency is ', self.disp_eff
        slick_area = 'WHAT??'

        if not isinstance(time_step, datetime.timedelta):
            time_step = datetime.timedelta(seconds=time_step)

        self._time_remaining = datetime.timedelta(seconds=time_step.total_seconds())
        zero = datetime.timedelta(seconds=0)
        if self.cur_state is None:
            # This is first step., setup inactivity if necessary
            if self.next_interval_index != 0:
                raise ValueError('disperse time series begins before time of first step!')
            else:
                self.cur_state = 'retired'
            print('start up')

        if self.cur_state == 'deactivated':
            # do deactivated stuff
            return

        while self._time_remaining > zero:

            if self.disp_eff == 0:
                #special case to shut operation down when dispersant is ineffective
                if self.cur_state == 'disperse' or self.cur_state == 'en_route':
                    self.report.append((model_time, 'Dispersant less than 20% efficient due to oil or environmental conditions. Returning to base'))
                    print self.report[-1]
                    o_w_t_t = datetime.timedelta(seconds=self.platform.one_way_transit_time(self.transit, payload=False))
                    self._op_start = self._op_end = None
                    self.cur_state = 'rtb'
                    if 'disperse' in self.cur_state:
                        self._next_state_time = model_time + o_w_t_t
                    else:
                        return_time = o_w_t_t.total_seconds() - (self._next_state_time - model_time).total_seconds()

                        self._next_state_time = model_time + datetime.timedelta(seconds=return_time)
                elif self.cur_state in ['retired', 'refuel_reload', 'ready', 'inactive']:
                    break

            ttni = self.time_to_next_interval(model_time)

            if ttni is None:
                if self.cur_state not in ['retired', 'reload', 'ready']:
                    raise ValueError('Operation is being deactivated while platform is active!')
                self.cur_state = 'deactivated'
                self.report.append((model_time, 'Disperse operation has ended and is deactivated'))
                print self.report[-1]
                break

            if self.cur_state == 'retired':
                if self.index_of(model_time) > -1 and self.timeseries[self.index_of(model_time)][0] == model_time:
                    #landed right on interval start, so ready immediately
                    self.cur_state = 'ready'
                    self.report.append((model_time, 'Begin new operational period'))
                    print self.report[-1]
                    continue
                self._time_remaining -= min(self._time_remaining, ttni)
                if self._time_remaining > zero:
                    model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                    # hit interval boundary before ending timestep.
                    # If ending current interval or no remaining time, do nothing
                    # if start of next interval, set state to 'ready'
                        # entering new operational interval
                        # ending current interval
                    if self.index_of(model_time) > -1:
                        self.cur_state = 'ready'
                        self.report.append((model_time, 'Begin new operational period'))
                        print self.report[-1]
                    else:
                        interval_idx = self.index_of(model_time - time_step + self._time_remaining)
                        self.report.append((model_time, 'Ending current operational period'))
                        print self.report[-1]

            elif self.cur_state == 'ready':
                if self.platform.sortie_possible(ttni, self.transit, self.pass_length):
                    # sortie is possible, so start immediately
                    self.report.append((model_time, 'Starting sortie'))
                    print self.report[-1]
                    self._next_state_time = model_time + datetime.timedelta(seconds=self.platform.one_way_transit_time(self.transit))
                    self.cur_state = 'en_route'
                    self._area_sprayed_this_sortie = 0
                    self._area_sprayed_this_ts = 0
                else:
                    # cannot sortie, so retire until next interval
                    self.cur_state = 'retired'
                    self.report.append((model_time, 'Retiring due to insufficient time remaining to conduct sortie'))
                    print self.report[-1]
                    self._time_remaining -= min(self._time_remaining, ttni)
                    model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)

            elif self.cur_state == 'en_route':
                time_left = self._next_state_time - model_time
                self._time_remaining -= min(self._time_remaining, time_left)
                model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                if self._time_remaining > zero:
                    self.report.append((model_time, 'Reached slick'))
                    print self.report[-1]
                    self._op_start = model_time
                    self._op_end = model_time + datetime.timedelta(seconds=self.platform.max_onsite_time(self.transit, self.loading_type))
                    self._cur_pass_num = 1
                    self.cur_state = 'disperse_' + str(self._cur_pass_num)
                    dur = datetime.timedelta(seconds=self.platform.pass_duration(self.pass_length))
                    self._next_state_time = model_time + dur
                    self.report.append((model_time, 'Starting disperse pass ' + str(self._cur_pass_num)))
                    print self.report[-1]

            elif 'disperse' in self.cur_state:
                pass_dur = datetime.timedelta(seconds=self.platform.pass_duration(self.pass_length))
                time_left_in_pass = self._next_state_time - model_time
                spray_time = min(self._time_remaining, time_left_in_pass)
                disp_possible = spray_time.total_seconds() * self.platform.eff_pump_rate(self.dosage)
                disp_actual = min(self._remaining_dispersant, disp_possible)
                oil_avail = self.dispersable_oil_amount(sc, 'm^3')
                treated_possible = disp_actual * self.disp_eff * self.disp_oil_ratio
                print 'treated_possible', treated_possible, 'disp_eff', self.disp_eff, 'ratio', self.disp_oil_ratio
                area_sprayed = disp_actual / self._dosage_m
                print ('oil sprayable', oil_avail)
                if self._remaining_dispersant == disp_actual:
                    # out of dispersant early, so short circuit into RTB
                    self._remaining_dispersant -= disp_actual
                    self._disp_sprayed_this_timestep += disp_actual
                    remainder_spray_time = datetime.timedelta(seconds=disp_actual / self.platform.eff_pump_rate(self.dosage))
                    self.oil_treated_this_timestep += min(treated_possible, self.dispersable_oil_amount(sc, 'm^3') - self.oil_treated_this_timestep)
                    self._time_remaining -= remainder_spray_time
                    self.model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                    self.report.append((model_time, 'Out of dispersant, returning to base'))
                    print self.report[-1]
                    o_w_t_t = datetime.timedelta(seconds=self.platform.one_way_transit_time(self.transit, payload=False))
                    self._next_state_time = model_time + o_w_t_t
                    self._op_start = self._op_end = None
                    self.cur_state = 'rtb'
                    continue

                elif oil_avail < treated_possible:
                    self.report.append((model_time, 'Treated all available oil, returning to base'))
                    print self.report[-1]
                    disp_to_treat = oil_avail / self.disp_oil_ratio / self.disp_eff
                    if disp_to_treat > disp_actual:
                        1 / 0
                    self._disp_sprayed_this_timestep += disp_to_treat
                    remainder_spray_time = datetime.timedelta(seconds=disp_actual / self.platform.eff_pump_rate(self.dosage))
                    self._time_remaining -= remainder_spray_time
                    self._remaining_dispersant -= self._disp_sprayed_this_timestep
                    self.oil_treated_this_timestep += min(treated_possible, self.dispersable_oil_amount(sc, 'm^3') - self.oil_treated_this_timestep)
                    model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                    o_w_t_t = datetime.timedelta(seconds=self.platform.one_way_transit_time(self.transit, payload=False))
                    self._next_state_time = model_time + o_w_t_t
                    self._op_start = self._op_end = None
                    self.cur_state = 'rtb'
                    continue

                else:
                    # spent entire remaining time spraying.
                    self._time_remaining -= min(self._time_remaining, time_left_in_pass)
                    self._remaining_dispersant -= disp_actual
                    self._disp_sprayed_this_timestep += disp_actual
                    self.oil_treated_this_timestep += min(treated_possible, self.dispersable_oil_amount(sc, 'm^3') - self.oil_treated_this_timestep)
                # ~
                # INSERT DISPERSION OF OIL HERE
                # ~
                model_time, time_step = self.update_time(self._time_remaining , model_time, time_step)
                if self._time_remaining > zero:
                    # completed a pass, so start the next one if possible
                    self.report.append((model_time, 'Disperse pass ' + str(self._cur_pass_num) + ' completed'))
                    print self.report[-1]
                    if self.platform.num_passes_possible(self._op_end - model_time, self.pass_length) > 0:
                        # can still make passes
                        self._cur_pass_num += 1
                        self.report.append((model_time, 'Starting disperse pass ' + str(self._cur_pass_num)))
                        print self.report[-1]
                        self.cur_state = 'disperse_' + str(self._cur_pass_num)
                        self._next_state_time = model_time + pass_dur
                    else:
                        # no passes possible, so RTB
                        self.report.append((model_time, 'No time for further passes, returning to base'))
                        print self.report[-1]
                        o_w_t_t = datetime.timedelta(seconds=self.platform.one_way_transit_time(self.transit, payload=False))
                        self._next_state_time = model_time + o_w_t_t
                        self._op_start = self._op_end = None
                        self._cur_pass_num = 1
                        self.cur_state = 'rtb'

            elif self.cur_state == 'rtb':
                time_left = self._next_state_time - model_time
                self._time_remaining -= min(self._time_remaining, time_left)
                model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                if self._time_remaining > zero:
                    self.report.append((model_time, 'Returned to base'))
                    print self.report[-1]
                    refuel_reload = datetime.timedelta(seconds=self.platform.refuel_reload(simul=self.loading_type))
                    self._next_state_time = model_time + refuel_reload
                    self.cur_state = 'refuel_reload'

            elif self.cur_state == 'refuel_reload':
                time_left = self._next_state_time - model_time
                self._time_remaining -= min(self._time_remaining, time_left)
                model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                if self._time_remaining > zero:
                    self.report.append((model_time, 'Refuel/reload complete'))
                    print self.report[-1]
                    self._remaining_dispersant = self.platform.get('payload', 'm^3')
                    self.cur_state = 'ready'

            elif self.cur_state == 'cascade':
                if self._next_state_time is None:
                    self._next_state_time = model_time + datetime.timedelta(seconds=self.platform.cascade_time(self.cascade_distance, payload=False))
                time_left = self._next_state_time - model_time
                self._time_remaining -= min(self._time_remaining, time_left)
                model_time, time_step = self.update_time(self._time_remaining, model_time, time_step)
                if self._time_remaining > zero:
                    self.report.append((model_time, 'Cascade complete'))
                    print self.report[-1]
                    self.cur_state = 'ready'
            else:
                raise ValueError('current state is not recognized: ' + self.cur_state)

    def update_time(self, time_remaining, model_time, time_step):
        if time_remaining > datetime.timedelta(seconds=0):
            return model_time + time_step - time_remaining, time_remaining
        else:
            return model_time, time_step

    def dispersable_oil_idxs(self, sc):
        # LEs must have a low viscosity, have not been fully chem dispersed, and must have a mass > 0
        idxs = np.where(sc['viscosity'] * 1000000 < 5000)[0]
        codes = sc['fate_status'][idxs] != bt_fate.disperse
        idxs = idxs[codes]
        nonzero_mass = sc['mass'][idxs] > 0
        idxs = idxs[nonzero_mass]
        return idxs

    def dispersable_oil_amount(self, sc, units='gal'):
        idxs = self.dispersable_oil_idxs(sc)
        if units in _valid_vol_units:
            tot_vol = np.sum(sc['mass'][idxs] / sc['density'][idxs])
            return max(0, uc.convert('volume', 'm^3', units, tot_vol))
        else:
            tot_mass = np.sum(sc['mass'][idxs])
            return max(0, tot_mass - self.oil_treated_this_timestep / np.mean(sc['density'][idxs]))

    def weather_elements(self, sc, time_step, model_time):
#         print 'disp sprayed', self._disp_sprayed_this_timestep
#         print 'disp remaining', self._remaining_dispersant
#         print 'oil treated', self.oil_treated_this_timestep

        elems_to_remove_from = self.dispersable_oil_idxs(sc)
        if self.oil_treated_this_timestep != 0:
            print 'ottt', self.oil_treated_this_timestep
            visc_eff_table = Disperse.visc_eff_table
#             disp_eff_per_le = [visc_eff_table[visc_eff_table.keys()[np.searchsorted(visc_eff_table.keys(), le)]] / 100 for le in sc['viscosity'][elems_to_remove_from] * 1000000]
#             proportions = disp_eff_per_le / np.mean(disp_eff_per_le)
            proportions = sc['mass'][elems_to_remove_from] / np.mean(sc['mass'][elems_to_remove_from])
            elem_densities = sc['density'][elems_to_remove_from]
            vol_reductions = proportions * self.oil_treated_this_timestep / len(elems_to_remove_from)
            mass_to_remove = vol_reductions * elem_densities  # oil_treated is in gallons, so need to change back to mass
            print 'indices', elems_to_remove_from
            print 'mass_to_remove', mass_to_remove
            self._remove_mass_indices(sc, mass_to_remove, elems_to_remove_from)
            sc.mass_balance['dispersed'] += sum(mass_to_remove)
            zero_or_disp = np.isclose(sc['mass'][elems_to_remove_from], 0)
            new_status = sc['fate_status'][elems_to_remove_from]
            new_status[zero_or_disp] = bt_fate.disperse
            sc['fate_status'][elems_to_remove_from] = new_status
            self.oil_treated_this_timestep = 0
            self.disp_sprayed_this_timestep = 0
            print sc['mass']



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
    timeseries = OnSceneTimeSeriesSchema()
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
                 'speed': 'kts'}

    _units_type = {'offset': ('length', _valid_dist_units),
                   'boom_length': ('length', _valid_dist_units),
                   'boom_draft': ('length', _valid_dist_units),
                   'speed': ('velocity', _valid_vel_units)}

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
            self._boom_capacity_remaining -= self.collected
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
                self._remove_mass_simple(data, self._ts_collected)

                self.logger.debug('{0} amount boomed for {1}: {2}'
                                  .format(self._pid, substance.name, self._ts_collected))

            if self._ts_burned:
                sc.mass_balance['burned'] += self._ts_burned
                sc.mass_balance['boomed'] -= self._ts_burned

class SkimUnitsSchema(MappingSchema):
    storage = SchemaNode(String(),
                         description='SI units for onboard storage',
                         validator=OneOf(_valid_vol_units))

    decant_pump = SchemaNode(String(),
                             description='SI units for decant',
                             validator=OneOf(_valid_dis_units))

    nameplate_pump = SchemaNode(String(),
                             description='SI units for nameplate',
                             validator=OneOf(_valid_dis_units))

    discharge_pump = SchemaNode(String(),
                             description='SI units for discharge',
                             validator=OneOf(_valid_dis_units))

    speed = SchemaNode(String(),
                       description='SI units for speed',
                       validator=OneOf(_valid_vel_units))

    swath_width = SchemaNode(String(),
                             description='SI units for length',
                             validator=OneOf(_valid_dist_units))

class SkimSchema(ResponseSchema):
    units = SkimUnitsSchema()
    speed = SchemaNode(Float())
    storage = SchemaNode(Float())
    swath_width = SchemaNode(Float())
    group = SchemaNode(String())
    throughput = SchemaNode(Float())
    nameplate_pump = SchemaNode(Float())
    skim_efficiency_type = SchemaNode(String())
    decant = SchemaNode(Float())
    decant_pump = SchemaNode(Float())
    rig_time = SchemaNode(TimeDelta())
    transit_time = SchemaNode(TimeDelta())
    offload_to = SchemaNode(String(), missing=drop)
    discharge_pump = SchemaNode(Float())
    recovery = SchemaNode(String())
    recovery_ef = SchemaNode(Float())
    timeseries = OnSceneTimeSeriesSchema()
    barge_arrival = SchemaNode(LocalDateTime(),
                               validator=validators.convertible_to_seconds,
                               missing=drop)


class Skim(Response):
    _state = copy.deepcopy(Response._state)
    _state += [Field('units', save=True, update=True),
               Field('speed', save=True, update=True),
               Field('storage', save=True, update=True),
               Field('swath_width', save=True, update=True),
               Field('group', save=True, update=True),
               Field('throughput', save=True, update=True),
               Field('nameplate_pump', save=True, update=True),
               Field('discharge_pump', save=True, update=True),
               Field('skim_efficiency_type', save=True, update=True),
               Field('decant', save=True, update=True),
               Field('decant_pump', save=True, update=True),
               Field('rig_time', save=True, update=True),
               Field('transit_time', save=True, update=True),
               Field('recovery', save=True, update=True),
               Field('recovery_ef', save=True, update=True)]

    _schema = SkimSchema

    _si_units = {'storage': 'bbl',
                 'decant_pump': 'gpm',
                 'nameplate_pump': 'gpm',
                 'speed': 'kts',
                 'swath_width': 'ft',
                 'discharge_pump': 'gpm'}

    _units_types = {'storage': ('storage', _valid_vol_units),
                    'decant_pump': ('decant_pump', _valid_dis_units),
                    'nameplate_pump': ('nameplate_pump', _valid_dis_units),
                    'speed': ('speed', _valid_vel_units),
                    'swath_width': ('swath_width', _valid_dist_units),
                    'discharge_pump': ('discharge_pump', _valid_dis_units)}

    def __init__(self,
                 speed,
                 storage,
                 swath_width,
                 group,
                 throughput,
                 nameplate_pump,
                 skim_efficiency_type,
                 recovery,
                 recovery_ef,
                 decant,
                 decant_pump,
                 discharge_pump,
                 rig_time,
                 transit_time,
                 units=_si_units,
                 **kwargs):

        super(Skim, self).__init__(**kwargs)

        self.speed = speed
        self.storage = storage
        self.swath_width = swath_width
        self.group = group
        self.throughput = throughput
        self.nameplate_pump = nameplate_pump
        self.recovery = recovery
        self.recovery_ef = recovery_ef
        self.decant = decant
        self.decant_pump = decant_pump
        self.rig_time = rig_time
        self.discharge_pump = discharge_pump
        self.skim_efficiency_type = skim_efficiency_type
        self.transit_time = transit_time
        self._units = dict(self._si_units)

        self._is_collecting = False
        self._is_transiting = False
        self._is_offloading = False
        self._is_rig_deriging = False

    def prepare_for_model_run(self, sc):
        self._setup_report(sc)
        self._storage_remaining = self.storage
        self._coverage_rate = self.swath_width * self.speed * 0.00233
        self.offload = (self.storage * 42 / self.discharge_pump) * 60

        if self.on:
            sc.mass_balance['skimmed'] = 0.0
            sc.mass_balance[self.id] = {'fluid_collected': 0.0,
                                        'emulsion_collected': 0.0,
                                        'oil_collected': 0.0,
                                        'water_collected': 0.0,
                                        'water_decanted': 0.0,
                                        'water_retained': 0.0,
                                        'area_covered': 0.0,
                                        'storage_remaining': 0.0}

        self._is_collecting = True

    def prepare_for_model_step(self, sc, time_step, model_time):
        if self._is_active(model_time, time_step):
            self._active = True
        else :
            self._active = False

        if not self.active:
            return

        self._time_remaining = time_step

        if type(self.barge_arrival) is datetime.date:
            # if there's a barge so a modified cycle
            while self._time_remaining > 0.:
                if self._is_collecting:
                    self._collect(sc, time_step, model_time)
        else:
            while self_time_remaining > 0.:
                if self._is_collecting:
                    self._collect(sc, time_step, model_time)

                if self._is_transiting:
                    self._transit(sc, time_step, model_time)

                if self._is_offloading:
                    self._offload(sc, time_step, model_time)


    def _collect(self, sc, time_step, model_time):
        thickness = self._get_thickness(sc)
        self._maximum_effective_swath = self.nameplate_pump * self.recovery / (63.13 * self.speed * thickness * self.throughput)

        if self.swath > self._maximum_effective_swath:
            swath = self._maximum_effective_swath;

        if swath > 1000:
            self.report.append('Swaths > 1000 feet may not be achievable in the field.')

        encounter_rate = thickness * self.speed * swath * 63.13
        rate_of_coverage = swath * self.speed * 0.00233

        if encounter_rate > 0:
            recovery = self._getRecoveryEfficiency()

            if recovery > 0:
                totalFluidRecoveryRate = encounter_rate * (self.throughput / recovery)

                if totalFluidRecoveryRate > self.nameplate_pump:
                    # total fluid recovery rate is greater than nameplate
                    # pump, recalculate the throughput efficiency and
                    # total fluid recovery rate again with the new throughput
                    throughput = self.nameplate_pump * recovery / encounter_rate
                    totalFluidRecoveryRate = encounter_rate * (throughput / recovery)
                    msg = ('{0.name} - Total Fluid Recovery Rate is greater than Nameplate \
                            Pump Rate, recalculating Throughput Efficiency').format(self)
                    self.logger.warni(msg)

                if throughput > 0:
                    emulsionRecoveryRate = encounter_rate * throughput

                    waterRecoveryRate = (1 - recovery) * totalFluidRecoveryRate
                    waterRetainedRate = waterRecoveryRate * (1 - self.decant)
                    computedDecantRate = (totalFluidRecoveryRate - emulsionRecoveryRate) * self.decant

                    decantRateDifference = 0.
                    if computedDecantRate > self.decant_pump:
                        decantRateDifference = computedDecantRate - self.decant_pump

                    recoveryRate = emulsionRecoveryRate + waterRecoveryRate
                    retainRate = emulsionRecoveryRate + weaterRetainedRate + decantRateDifference
                    oilRecoveryRate = emlusionRecoveryRate * (1 - sc['frac_water'].mean())

                    freeWaterRecoveryRate = recoveryRate - emulsionRecoveryRate
                    freeWaterRetainedRate = retainRate - emulsionRecoveryRate
                    freeWaterDecantRate = freeWaterRecoveryRate - freeWaterRetainedRate

                    timeToFill = .7 * self._storage_remaining / retainRate * 60

                    if timeToFill * 60 > self._time_remaining:
                        # going to take more than this timestep to fill the storage
                        time_collecting = self._time_remaining
                        self._time_remaining = 0.
                    else:
                        # storage is filled during this timestep
                        time_collecting = timeToFill
                        self._time_remaining -= timeToFill
                        self._transit_remaining = self.transit
                        self._collecting = False
                        self._transiting = True

                    self._ts_fluid_collected = retainRate * time_collecting
                    self._ts_emulsion_collected = emulsionRecoveryRate * time_collecting
                    self._ts_oil_collected = oilRecoveryRate * time_collecting
                    self._ts_water_collected = freeWaterRecoveryRate * time_collecting
                    self._ts_water_decanted = freeWaterDecantRate * time_collecting
                    self._ts_water_retained = freeWaterRetainedRate * time_collecting
                    self._ts_area_covered = rate_of_coverage * time_collecting

                    self._storage_remaining -= uc.convert('Volume', 'gal', 'bbl', self._ts_fluid_collected)

    def _transit(self, sc, time_step, model_time):
        # transiting back to shore to offload
        if self._time_remaining > self._transit_remaining:
            self._time_remaining -= self._transit_remaining
            self._transit_remaining = 0.
            self._is_transiting = False
            if self._storage_remaining == 0.0:
                self._is_offloading = True
            else:
                self._is_collecting = True
            self._offload_remaining = self.offload + self.rig_time
        else:
            self._transit_remaining -= self._time_remaining
            self._time_remaining = 0.

    def _offload(self, sc, time_step, model_time):
        if self._time_remaining > self._offload_remaining:
            self._time_remaining -= self_ofload_remaining
            self._offload_remaining = 0.
            self._storage_remaining = self.storage
            self._offloading = False
            self._transiting = True
        else:
            self._offload_remaining -= self._time_remaining
            self._time_remaining = 0.

    def weather_elements(self, sc, time_step, model_time):
        '''
        Remove mass from each le equally for now, no flagging for now
        just make sure the mass is from floating oil.
        '''
        if not self.active or len(sc) == 0:
            return

        les = sc.itersubstancedata(self.array_types)
        for substance, data in les:
            if len(data['mass']) is 0:
                continue

            if self._ts_oil_collected:
                sc.mass_balance['skimmed'] += self._ts_oil_collected
                self._remove_mass_simple(data, amount)

                self.logger.debug('{0} amount boomed for {1}: {2}'
                                  .format(self._pid, substance.name, self._ts_collected))

                platform_balance = sc.mass_balance[self.id]
                platform_balance['fluid_collected'] += self._ts_fluid_collected
                platform_balance['emulsion_collected'] += self._ts_emulsion_collected
                platform_balance['oil_collected'] += self._ts_oil_collected
                platform_balance['water_collected'] += self._ts_water_collected
                platform_balance['water_retained'] += self._ts_water_retained
                platform_balance['water_decanted'] += self._ts_water_decanted
                platform_balance['area_covered'] += self._ts_area_covered
                platform_balance['storage_remaining'] += self._storage_remaining


    def _getRecoveryEfficiency(self):
        # scaffolding method
        # will eventually include logic for calculating
        # recovery efficiency based on wind and oil visc.

        return self.recovery

if __name__ == '__main__':
    print None
    d = Disperse(name = 'test')
    p = Platform(_name='Test Platform')
    import pprint as pp
    ser = p.serialize()
    pp.pprint(ser)
    deser = Platform.deserialize(ser)

    pp.pprint(deser)

    p2 = Platform.new_from_dict(deser)
    ser2 = p2.serialize()
    pp.pprint(ser2)

    print 'INCORRECT BELOW'

    for k, v in ser.items():
        if p2.serialize()[k] != v:
            print p2.serialize()[k]

    pass
