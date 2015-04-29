'''
    For Adios behavior with no trajectory processing and no map,
    we add beaching events manually.
    This class controls the manual addition of beaching events in our
    model run.
    It is modelled as a weathering process.
'''

import datetime
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
    value = BeachingTupleSchema(default=(datetime.datetime.now(), 0))

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


class Beaching(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('timeseries', save=True, update=True),
               Field('units', save=True, update=True), ]
    _schema = BeachingSchema

    def __init__(self,
                 name,
                 active_start,
                 units='m^2',
                 timeseries=None,
                 **kwargs):
        '''
            Initialization for the manual beaching events.
        '''
        if 'active_stop' in kwargs:
            # user cannot set 'active_stop'
            kwargs.pop('active_stop')

        super(Beaching, self).__init__(active_start=active_start,
                                       **kwargs)

        self.name = name
        self.units = units

        if timeseries is not None:
            if units is None:
                raise TypeError('Units must be provided with timeseries')

            self.timeseries = timeseries
            self.convert_to_internal_volume()

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
        pass

    def weather_elements(self, sc, time_step, model_time):
        'We do not perform any element beaching yet.'
        pass
