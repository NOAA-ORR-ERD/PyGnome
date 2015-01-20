'''
Basic tests for timeseries
'''
from datetime import datetime

import numpy as np
from pytest import raises

from gnome.basic_types import datetime_value_2d
from gnome.utilities.timeseries import Timeseries

from ..conftest import testdata

wind_file = testdata['timeseries']['wind_ts']


def test_exceptions(invalid_rq):
    """
    Test ValueError exception thrown if improper input arguments
    Test TypeError thrown if units are not given - so they are None
    """
    # valid timeseries for testing

    dtv = np.zeros((4, ), dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv.time = [datetime(2012, 11, 06, 20, 10 + i, 30,) for i in range(4)]
    dtv.value = (1, 0)

    # Following also raises ValueError. This gives invalid (r,theta) inputs
    # which are rejected by the transforms.r_theta_to_uv_wind method.
    # It tests the inner exception is correct

    with raises(ValueError):
        invalid_dtv_rq = np.zeros((len(invalid_rq['rq']), ),
                                  dtype=datetime_value_2d)
        invalid_dtv_rq['value'] = invalid_rq['rq']
        Timeseries(timeseries=invalid_dtv_rq, format='r-theta')

    # exception raised if datetime values are not in ascending order
    # or not unique

    with raises(ValueError):
        # not unique datetime values
        dtv_rq = np.zeros((2, ),
                          dtype=datetime_value_2d).view(dtype=np.recarray)
        (dtv_rq.value[0])[:] = (1, 0)
        (dtv_rq.value[1])[:] = (1, 10)
        Timeseries(timeseries=dtv_rq)

    with raises(ValueError):
        # not in ascending order
        dtv_rq = np.zeros((4, ),
                          dtype=datetime_value_2d).view(dtype=np.recarray)
        dtv_rq.value = (1, 0)
        dtv_rq.time[:len(dtv_rq) - 1] = [datetime(2012, 11, 06, 20, 10 + i, 30)
                                         for i in range(len(dtv_rq) - 1)]
        Timeseries(timeseries=dtv_rq)


def test_init(wind_timeseries):
    ''
    rq = wind_timeseries['rq']
    uv = wind_timeseries['uv']
    ts = Timeseries(rq, format='r-theta')
    assert np.all(ts.get_timeseries()['time'] == rq['time'])
    assert np.allclose(ts.get_timeseries(format='r-theta')['value'],
                       rq['value'],
                       atol=1e-10)
    assert np.allclose(ts.get_timeseries()['value'],
                       uv['value'],
                       atol=1e-10)


def test_set_timeseries_prop():
    '''
    following operation requires a numpy array
    '''
    ts = Timeseries(filename=wind_file)

    # Following is a 0-D array, make sure it gets
    # converted to a 1-D array correctly
    x = (datetime.now().replace(microsecond=0, second=0), (4, 5))
    ts.set_timeseries(x)
    assert ts.get_timeseries()['time'] == x[0]
    assert np.allclose(ts.get_timeseries()['value'], x[1], atol=1e-6)
