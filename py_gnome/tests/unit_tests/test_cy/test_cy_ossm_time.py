#!/usr/bin/env python
"""
Unit tests for CyOSSMTime class
"""
import os
import pickle

from pytest import raises

import numpy
np = numpy

# import basic_types and subsequently lib_gnome
from gnome import basic_types
from gnome.basic_types import ts_format, seconds, velocity_rec

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime

datadir = os.path.join(os.path.dirname(__file__), r"sample_data")


def test_exceptions():
    with raises(IOError):
        # bad path
        CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WNDX'),
                   file_contains=ts_format.magnitude_direction)

    with raises(ValueError):
        # no inputs
        CyOSSMTime()

        # insufficient input info
        CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'))

        # insufficient input info
        CyOSSMTime(filename=os.path.join(datadir,
                                         'WindDataFromGnome_BadUnits.WND'),
                   file_contains=ts_format.magnitude_direction)

    with raises(ValueError):
        # file_contains has wrong int type
        CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                   file_contains=0)


def test_init_units():
    """
    Test __init__
    - correct path
    Updated so the user units are read from filename
    """
    ossmT2 = CyOSSMTime(filename=os.path.join(datadir,
                                              'WindDataFromGnome.WND'),
                        file_contains=ts_format.magnitude_direction)
    assert ossmT2.user_units == 'knots'


class TestTimeSeriesInit:
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
        ossm = CyOSSMTime(timeseries=self.tval)
        t_val = ossm.timeseries

        assert ossm.user_units == 'undefined'  # meters/second
        msg = 'CyOSSMTime.get_time_value did not return expected numpy array'
        np.testing.assert_array_equal(t_val, self.tval, msg, 0)

    def test_get_time_value(self):
        ossm = CyOSSMTime(timeseries=self.tval)

        actual = np.array(self.tval['value'], dtype=velocity_rec)
        time = np.array(self.tval['time'], dtype=seconds)
        vel_rec = ossm.get_time_value(time)
        print vel_rec

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, msg, 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, msg, 0)


class TestGetTimeValues:
    """
    Test get_time_value method for CyOSSMTime
    """
    # sample data generated and stored via Gnome GUI
    ossmT = CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                       file_contains=ts_format.magnitude_direction)

    def test_get_time_value(self):
        """
        Test get_time_values method. It gets the time value pairs for the
        model times stored in the data filename.
        For each line in the data filename, the ReadTimeValues method creates
        one time value pair
        This test just gets the time series that was created from the filename.
        It then invokes get_time_value for times in the time series.
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.timeseries

        # print t_val
        # assert False
        actual = np.array(t_val['value'], dtype=velocity_rec)
        time = np.array(t_val['time'], dtype=seconds)

        vel_rec = self.ossmT.get_time_value(time)

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        # TODO: Figure out why following fails??
        # tol = 1e-3
        # np.testing.assert_allclose(vel_rec, actual, tol, tol,
        #                            msg.format('get_time_value', tol), 0)
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, msg, 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, msg, 0)

    def test__set_time_value_handle_none(self):
        """Check TypeError exception for private method"""
        try:
            self.ossmT._set_time_value_handle(None)
        except TypeError:
            assert True

    def test_timeseries(self):
        """
        test setting the timeseries using timeseries property
        """
        t_val = self.ossmT.timeseries

        print 't_val before:', t_val['value']
        # need to learn how to do the following in 1 line of code
        t_val['value']['u'] += 2
        t_val['value']['v'] += 2
        print 't_val after:', t_val['value']

        self.ossmT.timeseries = t_val
        new_val = self.ossmT.timeseries

        tol = 1e-10
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        np.testing.assert_allclose(t_val['time'], new_val['time'],
                                   tol, tol, msg, 0)
        np.testing.assert_allclose(t_val['value']['u'], new_val['value']['u'],
                                   tol, tol, msg, 0)
        np.testing.assert_allclose(t_val['value']['v'], new_val['value']['v'],
                                   tol, tol, msg, 0)


class TestReadFileWithConstantWind:
    """
    Read contents for a filename that contains a constant wind.
    This will be just 1 line in the text filename.
    """
    file_name = os.path.join(datadir, 'WindDataFromGnomeConstantWind.WND')
    ossmT = CyOSSMTime(filename=file_name,
                       file_contains=ts_format.magnitude_direction)

    def test_get_time_value(self):
        """ Test get_time_values method. It gets the time value pair
            for the constant wind per the data filename.
            This test just gets the time value pair that was created
            from the filename. It then invokes get_time_value for that time
            in the time series and also looks at the velocity 100 sec later.
        Since wind is constant, the value should be unchanged
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.timeseries

        actual = np.array(t_val['value'], dtype=velocity_rec)
        time = np.array(t_val['time'] + (0, 100), dtype=seconds)

        vel_rec = self.ossmT.get_time_value(time)

        tol = 1e-6
        msg = ('{0} is not within a tolerance of '
               '{1}'.format('get_time_value', tol))
        for vel in vel_rec:
            np.testing.assert_allclose(vel['u'], actual['u'], tol, tol,
                                       msg, 0)
            np.testing.assert_allclose(vel['v'], actual['v'], tol, tol,
                                       msg, 0)


class TestObjectSerialization:
    '''
        Test all the serialization and deserialization methods that are
        available to the CyOSSMTime object.
    '''
    ossmT = CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                       file_contains=ts_format.magnitude_direction)

    def test_repr(self):
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

#        print "repr of ossmT"
#        print repr(self.ossmT)
        new_ossm = eval(repr(self.ossmT))

        assert new_ossm == self.ossmT
        assert repr(new_ossm) == repr(self.ossmT)

    def test_pickle(self):
        '''
            Test that the object can be pickled and unpickled
        '''
        new_ossm = pickle.loads(pickle.dumps(self.ossmT))
        assert new_ossm == self.ossmT
        assert repr(new_ossm) == repr(self.ossmT)


if __name__ == '__main__':
    # tt = TestTimeSeriesInit()
    # tt.test_init_timeseries()
    # tt.test_get_time_value()
    T = TestObjectSerialization()
    T.test_repr()