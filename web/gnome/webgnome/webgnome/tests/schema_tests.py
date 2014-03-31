import colander
import datetime

from dateutil import parser

from gnome.environment.wind import WindTimeSeriesSchema

from pytz import timezone
from unittest import TestCase
from webgnome.schema import WindSchema


class WindValueSchemaTests(TestCase):
    # JS: 12/11
    # It isn't clear how the following should work so comment the test for now
    # See ticket: https://trac.orr.noaa.gov/trac/GNOME-ADIOS/ticket/431
#==============================================================================
#     def test_serialize_strips_timezone(self):
#         # note -- serialize expects to receive a (datetime, (float, float))
#         # structure and outputs a (datetime, float, float) structure.
#         dt = datetime.datetime.now(timezone('US/Pacific'))
#         dt_without_tz = dt.replace(tzinfo=None)
#         data = [[numpy.datetime64(dt.isoformat()), [10.5, 180.5]]]
#         wind_value = WindTimeSeriesSchema().serialize(data)
#
#         self.assertEqual(wind_value[0][0], dt_without_tz.isoformat())
#         self.assertEqual(wind_value[0][1], data[1])
#         self.assertEqual(wind_value[0][2], data[2])
#==============================================================================

    def test_deserialize_strips_timezone(self):
        # note -- deserialize expects to receive a (datetime, float, float)
        # structure and outputs a (datetime, (float, float)) structure.
        dt = datetime.datetime(year=2013, month=2, day=5,
                               tzinfo=timezone('US/Pacific'))
        dt_without_tz = dt.replace(tzinfo=None)

        data = [[dt.isoformat(), '10.5', '180.5']]
        wind_value = WindTimeSeriesSchema().deserialize(data)

        self.assertEqual(wind_value[0][0].astype(object).isoformat(),
                         dt_without_tz.isoformat())
        self.assertEqual(wind_value[0][1][0], float(data[0][1]))
        self.assertEqual(wind_value[0][1][1], float(data[0][2]))

    def test_speed_must_be_nonzero(self):
        data = [datetime.datetime.now(), 0, 180.5]
        self.assertRaises(colander.Invalid,
                          WindTimeSeriesSchema().deserialize,
                          data)

    def test_direction_must_be_valid_degrees_true(self):
        data = [datetime.datetime.now(), 24, -1]
        self.assertRaises(colander.Invalid,
                          WindTimeSeriesSchema().deserialize,
                          data)

        data = [datetime.datetime.now(), 24, 361]
        self.assertRaises(colander.Invalid,
                          WindTimeSeriesSchema().deserialize,
                          data)


class WindSchemaTests(TestCase):

    def get_test_data(self):
        timeseries = [
            [datetime.datetime(year=2013, month=1, day=1,
                               hour=12, minute=30,
                               tzinfo=timezone('US/Pacific')).isoformat(),
             20.5,
             300.5],
            [datetime.datetime(year=2013, month=2, day=1,
                               hour=1, minute=30,
                               tzinfo=timezone('US/Pacific')).isoformat(),
             20.5,
             300.5],
        ]

        return {'timeseries': timeseries,
                'units': 'ft/hr'
        }

    def test_serialize(self):
        data = self.get_test_data()
        data = WindSchema().deserialize(data)
        serialized_wind = WindSchema().serialize(data)

        for idx, wind_value in enumerate(serialized_wind['timeseries']):
            test_val = data['timeseries'][idx]

            self.assertEqual(str(wind_value[0]),
                             test_val[0].astype(object).isoformat())
            self.assertEqual(wind_value[1], test_val[1][0])
            self.assertEqual(wind_value[2], test_val[1][1])

    def test_deserialize(self):
        data = self.get_test_data()
        expected_datetimes = []

        for idx, wind_value in enumerate(data['timeseries']):
            # Save a timezone-unaware version so we can more easily test it as
            # the expected result after deserializing.
            dt_without_timezone = parser.parse(
                wind_value[0]).replace(tzinfo=None)
            expected_datetimes.append(dt_without_timezone)

            data['timeseries'][idx][0] = str(wind_value[0])
            data['timeseries'][idx][1] = str(wind_value[1])
            data['timeseries'][idx][2] = str(wind_value[2])

        wind = WindSchema().deserialize(data)

        for idx, wind_value in enumerate(wind['timeseries']):
            test_values = data['timeseries'][idx]
            wind_dt = wind_value[0].astype(object).isoformat()

            self.assertEqual(wind_dt, expected_datetimes[idx].isoformat())
            self.assertEqual(wind_value[1][0], float(test_values[1]))
            self.assertEqual(wind_value[1][1], float(test_values[2]))
