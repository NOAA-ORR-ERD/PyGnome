#!/usr/bin/env python
"""
Unit tests for CyOSSMTime class
"""
import os
import pickle

from pytest import raises
import pytest

import numpy
np = numpy

# import basic_types and subsequently lib_gnome
from gnome import basic_types
from gnome.basic_types import ts_format, seconds, velocity_rec

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime, CyTimeseries
from gnome.cy_gnome.cy_shio_time import CyShioTime

datadir = os.path.join(os.path.dirname(__file__), r"sample_data")


@pytest.mark.parametrize('obj', [CyOSSMTime, CyTimeseries])
def test_exceptions(obj):
    with raises(IOError):
        # bad path
        obj(filename=os.path.join(datadir, 'WindDataFromGnome.WNDX'),
            file_format=ts_format.magnitude_direction)

    with raises(ValueError):
        # insufficient input info
        obj(filename=os.path.join(datadir, 'WindDataFromGnome.WND'))

    with raises(ValueError):
        # insufficient input info
        obj(filename=os.path.join(datadir, 'WindDataFromGnome_BadUnits.WND'),
            file_format=ts_format.magnitude_direction)

    with raises(ValueError):
        # file_format has wrong int type
        obj(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
            file_format=0)


def test_properties():
    '''
    todo: need to check if CyOSSMTime is reading/setting properties from file
    '''
    f = os.path.join(datadir, 'WindDataFromGnome.WND')
    cy = CyOSSMTime(f, 5)
    assert cy.station == ''


@pytest.mark.parametrize('obj', [CyOSSMTime, CyTimeseries])
def test_init_units(obj):
    """
    Test __init__
    - correct path
    Updated so the user units are read from filename
    """
    ossmT2 = obj(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                 file_format=ts_format.magnitude_direction)
    assert ossmT2.user_units == 'knots'


class TestObjectSerialization:
    '''
        Test all the serialization and deserialization methods that are
        available to the CyOSSMTime object.
    '''
    @pytest.mark.parametrize('obj', [CyOSSMTime, CyTimeseries])
    def test_repr(self, obj):
        '''
            Test that the repr method produces a string capable of reproducing
            the object.
        '''
        # This works, but in order to properly eval the repr string, we need
        # the top level gnome module, as well as the numpy 'array' type in the
        # global name space.
        # So before we try it, we first need to import like this:
        import gnome
        from numpy import array

        ossmT = obj(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                    file_format=ts_format.magnitude_direction)

#        print "repr of ossmT"
#        print repr(ossmT)
        new_ossm = eval(repr(ossmT))

        assert new_ossm == ossmT
        assert repr(new_ossm) == repr(ossmT)

    @pytest.mark.parametrize('obj', [CyOSSMTime, CyTimeseries])
    def test_pickle(self, obj):
        '''
            Test that the object can be pickled and unpickled
        '''
        ossmT = obj(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                    file_format=ts_format.magnitude_direction)
        new_ossm = pickle.loads(pickle.dumps(ossmT))
        assert new_ossm == ossmT
        assert repr(new_ossm) == repr(ossmT)


'Tests for child CyTimeseries object'


class TestCyTimeseries:
    """
    Test __init__ method and the exceptions it throws for CyOSSMTime
    """
    tval = np.array([(0, (1, 2)), (1, (2, 3))],
                    dtype=basic_types.time_value_pair)

    def test_init_timeseries(self):
        """
        Sets the time series in OSSMTimeValue_c equal to the
        externally supplied numpy array containing time_value_pair data
        It then reads it back to make sure data was set correctly
        """
        ossm = CyTimeseries(timeseries=self.tval)
        t_val = ossm.timeseries

        assert ossm.user_units == 'undefined'  # meters/second
        msg = ('{0}().get_time_value() did not return expected '
               'numpy array').format(ossm.__class__.__name__)
        np.testing.assert_array_equal(t_val, self.tval, msg, 0)

    def test_get_time_value(self):
        ossm = CyTimeseries(timeseries=self.tval)

        actual = np.array(self.tval['value'], dtype=velocity_rec)
        time = np.array(self.tval['time'], dtype=seconds)
        vel_rec = ossm.get_time_value(time)
        print vel_rec

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, msg, 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, msg, 0)

    def test__set_time_value_handle_none(self):
        """Check TypeError exception for private method"""
        try:
            ossm = CyTimeseries(timeseries=self.tval)
            ossm._set_time_value_handle(None)
        except TypeError:
            assert True

    def test_timeseries(self):
        """
        test setting the timeseries using timeseries property
        """
        ossm = CyTimeseries(timeseries=self.tval)
        t_val = ossm.timeseries

        print 't_val before:', t_val['value']
        # need to learn how to do the following in 1 line of code
        t_val['value']['u'] += 2
        t_val['value']['v'] += 2
        print 't_val after:', t_val['value']

        ossm.timeseries = t_val
        new_val = ossm.timeseries

        tol = 1e-10
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        np.testing.assert_allclose(t_val['time'], new_val['time'],
                                   tol, tol, msg, 0)
        np.testing.assert_allclose(t_val['value']['u'], new_val['value']['u'],
                                   tol, tol, msg, 0)
        np.testing.assert_allclose(t_val['value']['v'], new_val['value']['v'],
                                   tol, tol, msg, 0)

    def test_get_time_value_array(self):
        """
        Test get_time_values method. It gets the time value pairs for the
        model times stored in the data filename.
        For each line in the data filename, the ReadTimeValues method creates
        one time value pair
        This test just gets the time series that was created from the filename.
        It then invokes get_time_value for times in the time series.
        """
        ossmT = CyTimeseries(filename=os.path.join(datadir,
                                                   'WindDataFromGnome.WND'),
                             file_format=ts_format.magnitude_direction)
        # Let's see what is stored in the Handle to expected result
        t_val = ossmT.timeseries

        # print t_val
        # assert False
        actual = np.array(t_val['value'], dtype=velocity_rec)
        time = np.array(t_val['time'], dtype=seconds)

        vel_rec = ossmT.get_time_value(time)

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        # TODO: Figure out why following fails??
        # tol = 1e-3
        # np.testing.assert_allclose(vel_rec, actual, tol, tol,
        #                            msg.format('get_time_value', tol), 0)
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, msg, 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, msg, 0)

    def test_readfile_constant_wind(self):
        """
        Read contents for a filename that contains a constant wind.
        This will be just 1 line in the text filename.
        Test get_time_values method. It gets the time value pair
        for the constant wind per the data filename.
        This test just gets the time value pair that was created
        from the filename. It then invokes get_time_value for that time
        in the time series and also looks at the velocity 100 sec later.
        Since wind is constant, the value should be unchanged
        """
        file_name = os.path.join(datadir, 'WindDataFromGnomeConstantWind.WND')
        ossmT = CyTimeseries(filename=file_name,
                             file_format=ts_format.magnitude_direction)
        # Let's see what is stored in the Handle to expected result
        t_val = ossmT.timeseries

        actual = np.array(t_val['value'], dtype=velocity_rec)
        time = np.array(t_val['time'] + (0, 100), dtype=seconds)

        vel_rec = ossmT.get_time_value(time)

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        for vel in vel_rec:
            np.testing.assert_allclose(vel['u'], actual['u'], tol, tol,
                                       msg, 0)
            np.testing.assert_allclose(vel['v'], actual['v'], tol, tol,
                                       msg, 0)

if __name__ == '__main__':
    # tt = TestTimeSeriesInit()
    # tt.test_init_timeseries()
    # tt.test_get_time_value()
    T = TestObjectSerialization()
    T.test_repr()