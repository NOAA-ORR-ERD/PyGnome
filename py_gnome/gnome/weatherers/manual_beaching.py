'''
    For Adios behavior with no trajectory processing and no map,
    we add beaching events manually.
    This class controls the manual addition of beaching events in our
    model run.
    It is modeled as a weathering process.
'''

from datetime import datetime

import numpy as np

import nucos as uc

from gnome.basic_types import datetime_value_1d
from gnome.weatherers import Weatherer
from gnome.utilities.inf_datetime import InfDateTime

from gnome.persist import (validators, base_schema, DefaultTupleSchema,
                           LocalDateTime, DatetimeValue1dArraySchema,
                           SchemaNode, drop, Float, String, Range)

from .core import WeathererSchema
from .cleanup import RemoveMass
from gnome.environment.environment import WaterSchema


class BeachingTupleSchema(DefaultTupleSchema):
    '''
    Schema for each tuple in TimeSeries list
    '''
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
                          default=base_schema.now,
                          validator=validators.convertible_to_seconds)
    amount = SchemaNode(Float(), default=0,
                        validator=Range(min=0,
                                        min_err='amount must be '
                                                'greater than or equal to 0'))


class BeachingTimeSeriesSchema(DatetimeValue1dArraySchema):
    '''
    Schema for list of Amount tuples, to make the amount timeseries
    '''
    value = BeachingTupleSchema(default=(datetime.now()
                                         .replace(second=0, microsecond=0),
                                         0))

    def validator(self, node, cstruct):
        '''
        validate the amount timeseries numpy array
        '''
        validators.no_duplicate_datetime(node, cstruct)
        validators.ascending_datetime(node, cstruct)


class BeachingSchema(WeathererSchema):
    '''
    validate data after deserialize, before it is given back to pyGnome's
    from_dict to set _state of object
    '''
    units = SchemaNode(String(), default='m^3', save=True, update=True)
    timeseries = BeachingTimeSeriesSchema(missing=drop, save=True, update=True)
    water = WaterSchema(save=True, update=True, save_reference=True)


class Beaching(RemoveMass, Weatherer):
    '''
    It isn't really a reponse/cleanup option; however, it works in the same
    manner in that Beaching removes mass at a user specified rate. Mixin the
    RemoveMass functionality.
    '''
    _schema = BeachingSchema

    def __init__(self,
                 active_range,
                 units='m^3',
                 timeseries=None,
                 water=None,
                 **kwargs):
        '''
        Initialization for the manual beaching events.

        :param timeseries: array containing the volume of oil beached at
            specified time. The time corresponds with end time of the beaching
            contains: [(t0, v0), (t1, v1), ..]
            Assumes the delta time (t1 - t0) is larger than model's time_step.

        .. note:: Assumes the model's
            time_step is smaller than the timeseries timestep, meaning the
        '''
        super(Beaching, self).__init__(active_range=active_range, **kwargs)

        self.water = water

        self._units = None
        self.units = units

        # store mass removal rate as kg/sec for manual beaching
        self._rate = None
        self._timeseries = None

        if timeseries is not None and len(timeseries) > 0:
            self.timeseries = timeseries

    @property
    def timeseries(self):
        if self._timeseries is not None:
            return self._timeseries[1:]

        return None

    @timeseries.setter
    def timeseries(self, value):
        '''
        1. convert value to numpy array with dtype=datetime_value_1d
        2. set timeseries and also sets active_stop = timeseries['time'][-1]
        '''
        value = np.asarray(value, dtype=datetime_value_1d)

        # prepends active start to _timeseries array. This is for convenience
        to_insert = np.zeros(1, dtype=datetime_value_1d)

        if self.active_range[0] != InfDateTime('-inf'):
            to_insert['time'][0] = np.datetime64(self.active_range[0])

        self._timeseries = np.insert(value, 0, to_insert)
        self.active_range = (self.active_range[0],
                             self._timeseries['time'][-1].astype(datetime))

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        '''
        set units if value is in valid_vol_units
        '''
        if value in self.valid_vol_units or value in self.valid_mass_units:
            self._units = value
        else:
            msg = ('{0} are not valid volume or mass units.'
                   ' Not updated').format(value)
            self.logger.warning(msg)

    def convert_to_internal_volume(self):
        data = self.timeseries['value']
        from_unit = self.units
        to_unit = 'cubic meter'

        if from_unit != to_unit:
            data[:, 0] = uc.convert('Volume', from_unit, to_unit, data[:, 0])
            self.units = to_unit

    def prepare_for_model_run(self, sc):
        '''
        Preparation of data arrays related to beaching
        '''
        if self.on:
            sc.mass_balance['observed_beached'] = 0.0
            # force rate to be recalculated in case anything changed
            self._rate = None

    def _remove_mass(self, time_step, model_time, substance):
        '''
        returns the mass to be removed over time interval:
            (model_time, model_time + time_step)

        .. note:: invoked by weather_elements only if object is active for
            the step.

        fixme: the conversion to mass should happen all at once when the
               timeseries is set. and it should use substance.standard_density

        '''
        if self._rate is None:
            # ensure active start < timeseries['time'][0]
            # timedelta64 seems to be in seconds
            dt = np.diff(self._timeseries['time']).astype(np.float64)

            # convert timeseries to 'kg'
            dv = self.timeseries['value']

            unit_type = uc.FindUnitTypes()[self.units]
            if unit_type == 'mass':
                dm = uc.convert('mass', self.units, 'kg', dv)
            elif unit_type == 'volume':
                # water_temp = self.water.get('temperature')
                # rho = substance.density_at_temp(water_temp)
                volume = uc.convert('volume', self.units, 'm^3', dv)
                dm = volume * substance.standard_density
            else:
                # fixme: shouldn't this be check earlier??
                raise ValueError("{} is not a valid unit for beached oil"
                                 .format(self.unit))
            self._rate = dm / dt

        # find rate for time interval (model_time, model_time + time_step)
        # function is called for model_time within active start and active stop
        # so following should always work
        # Expect the timestep to be much smaller than the delta time between
        # timeseries, however, let's not make this assumption since it can't be
        # enforced
        t_int = np.where(np.datetime64(model_time) >=
                         self._timeseries['time'])[0][-1]

        # Say the time for timeseries is given as follows:
        #    [t_o, t_1, t_2, ..]
        #
        # if time interval resides within a timeseries timeinterval,
        #     so model_time > t_int and
        #        model_time + dt < t_int; then rm_mass = dt * rate[t_int]
        #
        # if time interval straddles two rates,
        #     so model_time > t_int and
        #        model_time + dt > t_int;
        # then rm_mass = \
        #    (self._timeseries['time'][t_int + 1] - model_time) * rate[t_int] +
        #    (dt - self._timeseries['time'][t_int + 1] - model_time) *
        #     rate[t_int + 1]
        #
        # The logic will also handle the case where the time interval straddles
        # multiple rates. This is not expected but the logic should work.
        time_remain = time_step
        start_time = model_time
        rm_mass = 0.0
        while time_remain > 0:
            dt_for_curr_rate = \
                (self._timeseries['time'][t_int + 1].astype(datetime) -
                 start_time).total_seconds()
            dt = min(time_remain, dt_for_curr_rate)
            rm_mass += self._rate[t_int] * dt
            time_remain -= dt

            # update start_time and t_int
            t_int += 1
            start_time = self._timeseries['time'][t_int].astype(datetime)

        return rm_mass

    def weather_elements(self, sc, time_step, model_time):
        '''
        remove equal fraction of mass from each component.
        '''
        if not self.active or len(sc) == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) == 0:
                continue
            rm_mass = self._remove_mass(self._timestep, model_time, substance)
            rm_mass_frac = rm_mass / data['mass'].sum()
            if rm_mass_frac > 1.0:
                rm_mass_frac = 1.0
                rm_mass = data['mass'].sum()
                msg = ("Beaching() removing more mass than available at {0}".
                       format(model_time))
                self.logger.warning(msg)

            data['mass_components'] = ((1 - rm_mass_frac) *
                                       data['mass_components'])
            data['mass'] = data['mass_components'].sum(1)

            sc.mass_balance['observed_beached'] += rm_mass
            self.logger.debug('{0} amount observed_beached for {1}: {2}'
                              .format(self._pid, substance.name, rm_mass))

        sc.update_from_fatedataview()
