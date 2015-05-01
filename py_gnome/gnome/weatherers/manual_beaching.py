'''
    For Adios behavior with no trajectory processing and no map,
    we add beaching events manually.
    This class controls the manual addition of beaching events in our
    model run.
    It is modelled as a weathering process.
'''

from datetime import datetime, timedelta
import copy

import numpy
np = numpy

from colander import (SchemaNode, drop,
                      Float, String, Range)

import unit_conversion as uc

from gnome import basic_types
from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable, Field

from gnome.persist import validators, base_schema
from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue1dArraySchema)
from .core import WeathererSchema
from .cleanup import RemoveMass


class AmountTuple(DefaultTupleSchema):
    amount = SchemaNode(Float(),
                        default=0,
                        validator=Range(min=0,
                                        min_err='amount must be '
                                                'greater than or equal to 0'
                                        )
                        )


class BeachingTupleSchema(DefaultTupleSchema):
    '''
    Schema for each tuple in TimeSeries list
    '''
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
                          default=base_schema.now,
                          validator=validators.convertible_to_seconds)
    amount = AmountTuple()


class BeachingTimeSeriesSchema(DatetimeValue1dArraySchema):
    '''
    Schema for list of Amount tuples, to make the amount timeseries
    '''
    value = \
        BeachingTupleSchema(default=(datetime.now().replace(second=0,
                                                            microsecond=0), 0))

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
    units = SchemaNode(String(), default='m^3')

    timeseries = BeachingTimeSeriesSchema(missing=drop)


class Beaching(RemoveMass, Serializable):
    '''
    It isn't really a reponse/cleanup option; however, it works in the same
    manner in that Beaching removes mass at a user specified rate. Mixin the
    RemoveMass functionality.
    '''
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('timeseries', save=True, update=True),
               Field('units', save=True, update=True), ]
    _schema = BeachingSchema

    def __init__(self,
                 name,
                 active_start,
                 units='m^3',
                 timeseries=None,
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
        if 'active_stop' in kwargs:
            # user cannot set 'active_stop'. active_stop is automatically set
            # to be the last time in the timeseries range
            kwargs.pop('active_stop')

        super(Beaching, self).__init__(active_start=active_start,
                                       **kwargs)

        self.name = name
        self.units = units

        # store mass removal rate as kg/sec for manual beaching
        self._rate = None
        self._timeseries = None

        if timeseries is not None:
            self.timeseries = timeseries

#==============================================================================
#             if units is None:
#                 raise TypeError('Units must be provided with timeseries')
# 
#             self.timeseries = timeseries
#             self.convert_to_internal_volume()
#==============================================================================

    @property
    def timeseries(self):
        return self._timeseries[1:]

    @timeseries.setter
    def timeseries(self, value):
        '''
        '''
        # do checks to ensure data is good before setting
        # prepends active_start to _timeseries array. This is for convenience
        self._timeseries = np.insert(value, 0, self.active_start)
        self.active_stop = self.timeseries[-1].astype(datetime)

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
            sc.weathering_data['manual_beached'] = 0.0

    def _remove_mass(self, time_step, model_time, substance):
        '''
        returns the mass to be removed over time interval:
            (model_time, model_time + time_step)

        .. note:: invoked by weather_elements only if object is active for
            the step.
        '''
        if self._rate is None:
            # ensure active_start < timeseries['time'][0]
            # explicitly make sure d_t is in seconds
            dt = [dt.total_seconds() for dt in
                  np.diff(np.insert(self.timeseries['time'], 0,
                                    self.active_start))]
            # convert timeseries to 'kg'
            dv = np.diff(np.insert(self.timeseries['value'], 0, 0))
            dm = (uc.convert('Volume', self.units, 'm^3', dv) *
                  substance.get_density())
            self._rate = dm/dt

        # find rate for time interval (model_time, model_time + time_step)
        # function is called for model_time within active_start and active_stop
        # so following should always work
        t_int = np.where(model_time > self._timeseries['time'])[0][0]
        dt = min(time_step, (self._timeseries['time'][t_int + 1] -
                             model_time).total_seconds())
        rm_mass = self._rate[t_int] * dt

        if dt < time_step:
            # (model_time, model_time + time_step) like in an interval with two
            # removal rates
            rm_mass += self._rate[t_int + 1] * (time_step - dt)

        return rm_mass

    def weather_elements(self, sc, time_step, model_time):
        '''
        remove equal fraction of mass from each component.
        '''
        if not self.active or len(sc) == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) is 0:
                continue
            rm_mass = self._remove_mass(time_step, model_time, substance)
            rm_mass_frac = rm_mass / data['mass'].sum()
            data['mass_components'] = \
                (1 - rm_mass_frac) * data['mass_components']
            data['mass'] = data['mass_components'].sum(1)

            sc.weathering_data['manual_beached'] += rm_mass
            self.logger.debug(self._pid + 'amount manual_beached for {0}: {1}'.
                              format(substance.name, rm_mass))

        sc.update_from_fatedataview()
