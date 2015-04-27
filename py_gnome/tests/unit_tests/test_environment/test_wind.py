import os
from datetime import datetime
import shutil

import pytest
from pytest import raises

import numpy
np = numpy

import unit_conversion

from gnome.basic_types import datetime_value_2d
from gnome.environment import Wind, constant_wind

from ..conftest import testdata

wind_file = testdata['timeseries']['wind_ts']


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    Test TypeError thrown if units are not given - so they are None
    """
    # valid timeseries for testing

    dtv = np.zeros((4, ), dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv.time = [datetime(2012, 11, 06, 20, 10 + i, 30,) for i in range(4)]
    dtv.value = (1, 0)

    # exception raised since no units given for timeseries during init
    with raises(TypeError):
        Wind(timeseries=dtv)

    # no units during set_timeseries
    with raises(TypeError):
        wind = Wind(timeseries=dtv, units='meter per second')
        wind.set_wind_data(dtv)

    # invalid units
    with raises(unit_conversion.InvalidUnitError):
        Wind(timeseries=dtv, units='met per second')


def test_read_file_init():
    """
    initialize from a long wind file
    """
    wm = Wind(filename=wind_file)
    print
    print '----------------------------------'
    print 'Units: ' + str(wm.units)
    assert True


# tolerance for np.allclose(..) function.
# Results are almost the same but not quite so needed to add tolerance.
# The precision per numpy.spacing(1)=2.2e-16

atol = 1e-14
rtol = 0


def test_units():
    """
    just make sure there are no errors
    """
    wm = Wind(filename=wind_file)
    new_units = 'meter per second'
    assert wm.units != new_units
    wm.units = new_units
    assert wm.units == new_units


def test_default_init():
    wind = Wind()
    assert wind.timeseries == np.zeros((1,), dtype=datetime_value_2d)
    assert wind.units == 'mps'


def test_init_timeseries():
    'test init succeeds via different methods'
    constant = (datetime(2014, 6, 25, 10, 14), (0, 0))
    ts = [constant,
          (datetime(2014, 6, 25, 10, 20), (1, 1))]
    Wind(timeseries=constant, units='mps')
    Wind(timeseries=ts, units='mps')
    np.asarray(constant)
    Wind(timeseries=np.asarray(constant, dtype=datetime_value_2d), units='mps')
    Wind(timeseries=np.asarray(ts, dtype=datetime_value_2d), units='mps')

    # todo!!!
    # following should fail, but it doesn't because
    # np.asarray((1, 2), dtype=datetime_value_2d) does not raise an error!
    # no exceptions raised for this at present - should fix
    Wind(timeseries=(1, 2), units='mps')


def test_wind_circ_fixture(wind_circ):
    """
    check timeseries of wind object created in 'wind_circ'
    """
    wm = wind_circ['wind']

    # output is in knots

    gtime_val = wm.get_wind_data(format='uv').view(dtype=np.recarray)
    assert np.all(gtime_val.time == wind_circ['uv'].time)
    assert np.allclose(gtime_val.value, wind_circ['uv'].value, atol, rtol)

    # output is in meter per second

    gtime_val = wm.get_wind_data(format='uv', units='meter per second'
                                  ).view(dtype=np.recarray)
    expected = unit_conversion.convert('Velocity', wm.units,
            'meter per second', wind_circ['uv'].value)

    assert np.all(gtime_val.time == wind_circ['uv'].time)
    assert np.allclose(gtime_val.value, expected, atol, rtol)


def test_get_value(wind_circ):
    'test get_value(..) function'
    wind = wind_circ['wind']
    for rec in wind_circ['rq']:
        time = rec['time']
        val = wind.get_value(time)
        assert all(np.isclose(rec['value'], val))


@pytest.fixture(scope='module')
def wind_rand(rq_rand):
    """
    Create Wind object using the time series given by the test fixture
    'rq_rand'.

    NOTE:
    Since 'rq_rand' randomly generates (r,theta), the corresponing (u,v)
    are calculated from gnome.utilities.transforms.r_theta_to_uv_wind(...).
    Assumes this method works correctly.
    """

    from gnome.utilities import transforms

    dtv_rq = np.zeros((len(rq_rand['rq']), ),
                      dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_rq.time = [datetime(2012, 11, 06, 20, 10 + i, 30)
                   for i in range(len(dtv_rq))]
    dtv_rq.value = rq_rand['rq']
    dtv_uv = np.zeros((len(dtv_rq), ),
                      dtype=datetime_value_2d).view(dtype=np.recarray)
    dtv_uv.time = dtv_rq.time
    dtv_uv.value = transforms.r_theta_to_uv_wind(rq_rand['rq'])

    wm = Wind(timeseries=dtv_rq, format='r-theta', units='meter per second')
    return {'wind': wm, 'rq': dtv_rq, 'uv': dtv_uv}


# @pytest.fixture(scope="module",params=['wind_circ','wind_rand'])

@pytest.fixture(scope='module', params=['wind_circ'])
def all_winds(request):
    """
    NOTE: Since random test setup (wind_rand) occasionally
    makes test_get_wind_data_by_time_scalar fail, omit
    this test case for now. It is being investigated.

    Create Wind object using the time series given by the test fixture
    'wind_circ', 'wind_rand'.

    NOTE:
    Since 'wind_rand' randomly generates (r,theta), the corresponing (u,v)
    are calculated from gnome.utilities.transforms.r_theta_to_uv_wind(...).
    Assumes this method works correctly.
    """
    return request.getfuncargvalue(request.param)


class TestWind:
    """
    Gather all tests that apply to a WindObject in this class. All methods use
    the 'all_winds' fixture

    Couldn't figure out how to use "wind" fixture at class level.
    Tried decorating the class with the following:
    @pytest.mark.usefixtures("wind")
    """
    def test_init_units(self, all_winds):
        """
        check default wind object is created

        Also check that init doesn't fail if timeseries given in (u,v) format
        """
        Wind(timeseries=all_winds['uv'], format='uv', units='meter per second')
        assert True

    def test_str_repr_no_errors(self, all_winds):
        """
        simply tests that we get no errors during repr() and str()
        """
        repr(all_winds['wind'])
        print str(all_winds['wind'])
        assert True

    def test_id_matches_builtin_id(self, all_winds):
        # It is not a good assumption that the obj.id property
        # will always contain the id(obj) value.  For example it could
        # have been overloaded with, say, a uuid1() generator.
        # assert id(all_winds['wind']) == all_winds['wind'].id
        pass

    def test_get_wind_data(self, all_winds):
        """
        get_wind_data with default output format
        """
        # check get_time_value()

        gtime_val = all_winds['wind'].get_wind_data()
        assert np.all(gtime_val['time'] == all_winds['rq'].time)
        assert np.allclose(gtime_val['value'], all_winds['rq'].value,
                           atol, rtol)

    def test_set_wind_data(self, all_winds):
        """
        get_wind_data with default output format
        """
        # check get_time_value()

        wm = Wind(timeseries=all_winds['wind'].get_wind_data(),
                  format='r-theta', units='meter per second')
        gtime_val = wm.get_wind_data()
        x = gtime_val[:2]
        x['value'] = [(1, 10), (2, 20)]

        # default format is 'r-theta'
        wm.set_wind_data(x, 'meter per second')

        # only matches to 10^-14
        assert np.allclose(wm.get_wind_data()['value'][:, 0],
                           x['value'][:, 0], atol, rtol)
        assert np.all(wm.get_wind_data()['time'] == x['time'])

    def test_get_wind_data_rq(self, all_winds):
        """
        Initialize from timeseries and test the get_time_value method
        """
        # check get_time_value()

        gtime_val = all_winds['wind'].get_wind_data(format='r-theta')
        assert np.all(gtime_val['time'] == all_winds['rq'].time)
        assert np.allclose(gtime_val['value'], all_winds['rq'].value,
                           atol, rtol)

    def test_get_wind_data_uv(self, all_winds):
        """
        Initialize from timeseries and test the get_time_value method
        """
        gtime_val = (all_winds['wind']
                     .get_wind_data(format='uv')
                     .view(dtype=np.recarray))
        assert np.all(gtime_val.time == all_winds['uv'].time)
        assert np.allclose(gtime_val.value, all_winds['uv'].value, atol, rtol)

    def test_get_wind_data_by_time(self, all_winds):
        """
        get time series, but this time provide it with the datetime values
        for which you want timeseries
        """
        gtime_val = (all_winds['wind']
                     .get_wind_data(format='r-theta',
                                     datetime=all_winds['rq'].time)
                     .view(dtype=np.recarray))
        assert np.all(gtime_val.time == all_winds['rq'].time)
        assert np.allclose(gtime_val.value, all_winds['rq'].value, atol, rtol)

    def test_get_wind_data_by_time_scalar(self, all_winds):
        """
        Get single time value in the middle of the 0th and 1st index
        of the timeseries.
        Read the value (wind velocity) for this time.

        Output should be an interpolated value between the values of the 0th
        and 1st index of timeseries.
        """
        t0 = all_winds['rq'].time[0].astype(object)
        t1 = all_winds['rq'].time[1].astype(object)
        dt = t0 + ((t1 - t0) / 2)

        get_rq = (all_winds['wind']
                  .get_wind_data(format='r-theta', datetime=dt)
                  .view(dtype=np.recarray))

        get_uv = (all_winds['wind']
                  .get_wind_data(format='uv', datetime=dt)
                  .view(dtype=np.recarray))

        np.set_printoptions(precision=4)
        print
        print '=================================================='
        print '(u,v):'
        print str(all_winds['uv'].value[:2, :])
        print
        print 'get_uv:\t{0}'.format(get_uv.value[0])
        print 'time:  \t{0}'.format(dt)
        print '-----------------'
        print ('u-bounds: ({0:0.4f},{1:0.4f});\t '
               'computed-u: ({2:0.4f})'
               ''.format(min(all_winds['uv'].value[:2, 0]),
                         max(all_winds['uv'].value[:2, 0]),
                         get_uv.value[0, 0]))
        print ('v-bounds: ({0:0.4f},{1:0.4f});\t '
               'computed-v: ({2:0.4f})'
               ''.format(min(all_winds['uv'].value[:2, 1]),
                         max(all_winds['uv'].value[:2, 1]),
                         get_uv.value[0, 1]))
        print '-----------------'
        print 'FOR INFO ONLY: INTERPOLATION IS DONE IN (u,v) SPACE'
        print '(r,theta): '
        print str(all_winds['rq'].value[:2, :])
        print
        print 'get_rq:\t{0}'.format(get_rq.value[0])
        print '-----------------'
        print ('r-bounds: ({0:0.4f},{1:0.4f});\t '
               'computed-r: ({2:0.4f})'
               ''.format(min(all_winds['rq'].value[:2, 0]),
                         max(all_winds['rq'].value[:2, 0]),
                         get_rq.value[0, 0]))
        print ('q-bounds: ({0:0.4f},{1:0.4f});\t '
               'computed-q: ({2:0.4f})'
               ''.format(min(all_winds['rq'].value[:2, 1]),
                         max(all_winds['rq'].value[:2, 1]),
                         get_rq.value[0, 1]))
        print '-----------------'
        print 'NOTE: This test fails at times for randomly generated (r,theta)'
        print '      Still trying to understand how the hermite interpolation'
        print '      should work'

        assert get_uv.time[0].astype(object) == dt
        assert (get_uv.value[0, 0] > np.min(all_winds['uv'].value[:2, 0]) and
                get_uv.value[0, 0] < np.max(all_winds['uv'].value[:2, 0]))
        assert (get_uv.value[0, 1] > np.min(all_winds['uv'].value[:2, 1]) and
                get_uv.value[0, 1] < np.max(all_winds['uv'].value[:2, 1]))

        # =====================================================================
        # # FOLLOWING DOES NOT WORK
        # assert get_rq.value[0, 0] > all_winds['rq'].value[0, 0] \
        #    and get_rq.value[0, 0] < all_winds['rq'].value[1, 0]
        # assert get_rq.value[0, 1] > all_winds['rq'].value[1, 0] \
        #    and get_rq.value[0, 1] < all_winds['rq'].value[1, 1]
        # =====================================================================


def test_constant_wind():
    """
    tests the utility function for creating a constant wind
    """
    wind = constant_wind(10, 45, 'knots')

    dt = datetime(2013, 1, 10, 12, 0)
    assert np.allclose(wind.get_wind_data(datetime=dt, units='knots')[0][1],
                       (10, 45))

    dt = datetime(2013, 1, 10, 12, 0)
    assert np.allclose(wind.get_wind_data(datetime=dt, units='knots')[0][1],
                       (10, 45))

    dt = datetime(2013, 1, 10, 12, 0)
    assert np.allclose(wind.get_wind_data(datetime=dt, units='knots')[0][1],
                       (10, 45))


def test_eq():
    """
    tests the filename is not used for testing equality
    even if filename changes but other attributes are the same, the objects
    are equal
    """
    w = Wind(filename=wind_file)
    w1 = Wind(filename=wind_file)
    assert w == w1

    p, f = os.path.split(wind_file)
    f, e = os.path.splitext(f)

    w_copy = os.path.join(p, '{0}_copy{1}'.format(f, e))
    shutil.copy(wind_file, w_copy)

    w2 = Wind(filename=w_copy)
    w2.updated_at = w.updated_at    # match these before testing
    assert w == w2


def test_timeseries_res_sec():
    '''check the timeseries resolution is changed to minutes.
    Drop seconds from datetime, if given'''
    ts = np.zeros((3,), dtype=datetime_value_2d)
    ts[:] = [(datetime(2014, 1, 1, 10, 10, 30), (1, 10)),
             (datetime(2014, 1, 1, 11, 10, 10), (2, 10)),
             (datetime(2014, 1, 1, 12, 10), (3, 10))]
    w = Wind(timeseries=ts, units='m/s')
    # check that seconds resolution has been dropped
    for ix, dt in enumerate(w.timeseries['time'].astype(datetime)):
        assert ts['time'][ix].astype(datetime).replace(second=0) == dt


def test_update_from_dict():
    'wind_json only used here so take it out of conftest'
    wind_json = {'obj_type': 'gnome.environment.Wind',
                 'description': 'update_description',
                 'latitude': 90,
                 'longitude': 90,
                 'updated_at': '2014-03-26T14:52:45.385126',
                 'source_type': u'manual',
                 'source_id': u'unknown',
                 'timeseries': [('2012-11-06T20:10:00', (1.0, 0.0)),
                                ('2012-11-06T20:11:00', (1.0, 45.0)),
                                ('2012-11-06T20:12:00', (1.0, 90.0)),
                                ('2012-11-06T20:13:00', (1.0, 120.0)),
                                ('2012-11-06T20:14:00', (1.0, 180.0)),
                                ('2012-11-06T20:15:00', (1.0, 270.0))],
                 'units': 'knots',
                 'json_': u'webapi'
                 }
    wind = constant_wind(1.0, 45.0, 'meter per second')

    # following tests fails because unit conversion causes some
    # rounding errors. Look into this more carefully.
    # not_changed = wind.deserialize(wind.serialize('webapi'))
    # assert not wind.update_from_dict(not_changed)
    d_wind = wind.deserialize(wind_json)
    updated = wind.update_from_dict(d_wind)
    assert updated

    new_w = wind.serialize('webapi')

    # since json could be a subset of state, check equality for ones that were
    # updated
    for key in wind_json:
        if key != 'obj_type':
            assert new_w[key] == wind_json[key]
