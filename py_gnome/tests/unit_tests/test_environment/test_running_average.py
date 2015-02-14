import os
from datetime import datetime
import shutil
import json

import pytest
from pytest import raises

import numpy
np = numpy

import unit_conversion

from gnome.utilities.time_utils import date_to_sec
from gnome.basic_types import datetime_value_2d
from gnome.environment import Wind, constant_wind, RunningAverage

from ..conftest import testdata

data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
wind_file = testdata['timeseries']['wind_ts_av']

def test_av_from_variable_wind():
    '''
    test variable wind to running average
    '''
    ts = np.zeros((4,), dtype=datetime_value_2d)
    ts[:] = [(datetime(2012, 9, 7, 8, 0), (10, 270)),
             (datetime(2012, 9, 7, 14, 0), (28, 270)),
             (datetime(2012, 9, 7, 20, 0), (28, 270)),
             (datetime(2012, 9, 8, 02, 0), (10, 270))]
             
    wm = Wind(timeseries=ts, units='m/s')
    #wm = Wind(filename=wind_file)
    av = RunningAverage(wm)
#     print "wm.ossm.timeseries"
#     print wm.ossm.timeseries[:]
#     print "av.ossm.timeseries"
#     print av.ossm.timeseries[:]

    assert av.ossm.timeseries['time'][0] == wm.ossm.timeseries['time'][0]
    assert av.ossm.timeseries['value']['u'][0] == wm.ossm.timeseries['value']['u'][0]
    assert av.ossm.timeseries['value']['u'][1] == 10.5

def test_av_from_constant_wind():
    '''
    test constant wind gives constant running average
    '''
    wind = constant_wind(1., 270.)

    av = RunningAverage(wind)
#     print "wind.ossm.timeseries"
#     print wind.ossm.timeseries[:]
#     print "av.ossm.timeseries"
#     print av.ossm.timeseries[:]
    assert av.ossm.timeseries[0] == wind.ossm.timeseries[0]


def test_av_from_timeseries():
    '''
    doesn't do any unit or format conversion
    assume m/s and uv
    '''
    ts = np.zeros((3,), dtype=datetime_value_2d)
    ts[:] = [(datetime(2014, 1, 1, 10, 10, 00), (10, 0)),
             (datetime(2014, 1, 1, 11, 10, 00), (20, 0)),
             (datetime(2014, 1, 1, 12, 10), (10, 0))]
    av = RunningAverage(timeseries=ts)
    #print "av1"
    #print av.ossm.timeseries[:]
   
    
def test_full_run():
    '''
    test a wind series that has a constant average
    '''
    ts = np.zeros((10,), dtype=datetime_value_2d)
    ts[:] = [(datetime(2015, 1, 1, 1, 0), (10, 0)),
             (datetime(2015, 1, 1, 2, 0), (20, 0)),
             (datetime(2015, 1, 1, 3, 0), (10, 0)),
             (datetime(2015, 1, 1, 4, 0), (20, 0)),
             (datetime(2015, 1, 1, 5, 0), (10, 0)),
             (datetime(2015, 1, 1, 6, 0), (20, 0)),
             (datetime(2015, 1, 1, 7, 0), (10, 0)),
             (datetime(2015, 1, 1, 8, 0), (20, 0)),
             (datetime(2015, 1, 1, 9, 0), (10, 0)),
             (datetime(2015, 1, 1, 10, 0), (20, 0))]

    start_time = datetime(2015, 1, 1, 1)
    model_time = start_time

    running_av = RunningAverage(timeseries=ts)
    running_av.prepare_for_model_run(model_time)
    running_av.prepare_for_model_step(model_time)

    print "running_av"
    print running_av.ossm.timeseries[:]

    model_time = datetime(2015, 1, 1, 4, 0)
    running_av.prepare_for_model_run(model_time)
    running_av.prepare_for_model_step(model_time)

    print "running_av2"
    print running_av.ossm.timeseries[:]

    assert np.all(running_av.ossm.timeseries['value']['u'][3:9] == 15)


def test_full_run_extended():
    '''
    test algorithm resets two day running average with a new time
    '''
    ts = np.zeros((20,), dtype=datetime_value_2d)
    ts[:] = [(datetime(2015, 1, 1, 1, 0), (10, 0)),
             (datetime(2015, 1, 1, 2, 0), (20, 0)),
             (datetime(2015, 1, 1, 3, 0), (10, 0)),
             (datetime(2015, 1, 1, 4, 0), (20, 0)),
             (datetime(2015, 1, 1, 5, 0), (10, 0)),
             (datetime(2015, 1, 1, 6, 0), (20, 0)),
             (datetime(2015, 1, 1, 7, 0), (10, 0)),
             (datetime(2015, 1, 1, 8, 0), (20, 0)),
             (datetime(2015, 1, 1, 9, 0), (10, 0)),
             (datetime(2015, 1, 1, 10, 0), (20, 0)),
             (datetime(2015, 1, 5, 1, 0), (10, 0)),
             (datetime(2015, 1, 5, 2, 0), (20, 0)),
             (datetime(2015, 1, 5, 3, 0), (10, 0)),
             (datetime(2015, 1, 5, 4, 0), (20, 0)),
             (datetime(2015, 1, 5, 5, 0), (10, 0)),
             (datetime(2015, 1, 5, 6, 0), (20, 0)),
             (datetime(2015, 1, 5, 7, 0), (10, 0)),
             (datetime(2015, 1, 5, 8, 0), (20, 0)),
             (datetime(2015, 1, 5, 9, 0), (10, 0)),
             (datetime(2015, 1, 5, 10, 0), (20, 0))]

    wm = Wind(filename=wind_file, units='mps')

    start_time = datetime(2015, 1, 1, 1)
    model_time = start_time

    running_av = RunningAverage(wind=wm)
    #running_av = RunningAverage(timeseries=ts)
    running_av.prepare_for_model_run(model_time)
    running_av.prepare_for_model_step(model_time)

    print "running_av"
    print running_av.ossm.timeseries[:]

    model_time = datetime(2015, 1, 5, 4, 0)
    running_av.prepare_for_model_run(model_time)
    running_av.prepare_for_model_step(model_time)

    print "running_av2"
    print running_av.ossm.timeseries[:]

    assert np.all(running_av.ossm.timeseries['value']['u'][:] == 15)


def test_past_hours_to_average():
    """
    just make sure there are no errors
    """
    wm = Wind(filename=wind_file)
    av = RunningAverage(wm)
    assert av.past_hours_to_average == 3
    av.past_hours_to_average = 6
    assert av.past_hours_to_average == 6


def test_default_init():
    av = RunningAverage()
    assert av.timeseries == np.zeros((1,), dtype=datetime_value_2d)
    assert av.units == 'mps'


def test_serialize_deserialize():
    'test serialize/deserialize for webapi'
    wind = constant_wind(1., 0)
    av = RunningAverage(wind)
    json_ = av.serialize()
    json_['wind'] = wind.serialize()

    # deserialize and ensure the dict's are correct
    d_av = RunningAverage.deserialize(json_)
    assert d_av['wind'] == Wind.deserialize(json_['wind'])
    d_av['wind'] = wind
    av.update_from_dict(d_av)
    assert av.wind is wind
