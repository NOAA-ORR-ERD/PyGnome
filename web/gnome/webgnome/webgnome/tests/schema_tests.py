import colander
import datetime
import numpy

from unittest import TestCase
from pytz import timezone

from webgnome import schema


class WindValueSchemaTests(TestCase):
    def test_serialize_strips_timezone(self):
        dt = datetime.datetime.now(timezone('US/Pacific'))
        dt_without_tz = dt.replace(tzinfo=None)
        data = [dt, 10.5, 180.5]
        wind_value = schema.TimeseriesValueSchema().serialize(data)

        self.assertEqual(wind_value[0], dt_without_tz.isoformat())
        self.assertEqual(wind_value[1], str(data[1]))
        self.assertEqual(wind_value[2], str(data[2]))

    def test_deserialize_strips_timezone(self):
        dt = datetime.datetime.now(timezone('US/Pacific'))
        dt_without_tz = dt.replace(tzinfo=None)
        data = [dt.isoformat(), '10.5','180.5']
        wind_value = schema.TimeseriesValueSchema().deserialize(data)

        self.assertEqual(wind_value[0], dt_without_tz)
        self.assertEqual(wind_value[1], float(data[1]))
        self.assertEqual(wind_value[2], float(data[2]))

    def test_speed_must_be_nonzero(self):
        data = [datetime.datetime.now(), 0, 180.5]
        self.assertRaises(colander.Invalid,
                          schema.TimeseriesValueSchema().deserialize, data)


class WindSchemaTests(TestCase):

    def get_test_data(self):
        timeseries = [
            [
                datetime.datetime(year=2013, month=1, day=1,
                                              hour=12, minute=30,
                                              tzinfo=timezone('US/Pacific')),
                20.5,
                300.5
            ],
            [
                datetime.datetime(year=2013, month=2, day=1,
                                              hour=1, minute=30,
                                              tzinfo=timezone('US/Pacific')),
                20.5,
                300.5
            ],
        ]

        return {
            'timeseries': timeseries,
            'units': 'ft/hr'
        }

    def test_serialize(self):
        data = self.get_test_data()
        serialized_wind = schema.WindSchema().serialize(data)

        for idx, wind_value in enumerate(serialized_wind['timeseries']):
            test_val = data['timeseries'][idx]
            self.assertEqual(wind_value[0],
                             test_val[0].replace(tzinfo=None).isoformat())
            self.assertEqual(wind_value[1], str(test_val[1]))
            self.assertEqual(wind_value[2], str(test_val[2]))


    def test_deserialize(self):
        data = self.get_test_data()
        expected_datetimes = []

        for idx, wind_value in enumerate(data['timeseries']):
            # Save a timezone-unaware version so we can more easily test it as
            # the expected result after deserializing.
            expected_datetimes.append(str(wind_value[0].replace(tzinfo=None)))
            data['timeseries'][idx][0] = str(wind_value[0])
            data['timeseries'][idx][1] = str(wind_value[1])
            data['timeseries'][idx][2] = str(wind_value[2])

        wind = schema.WindSchema().deserialize(data)

        for idx, wind_value in enumerate(wind['timeseries']):
            test_values= data['timeseries'][idx]
            self.assertEqual(str(wind_value[0]),
                             str(numpy.datetime64(expected_datetimes[idx])))
            self.assertEqual(wind_value[1][0], float(test_values[1]))
            self.assertEqual(wind_value[1][1], float(test_values[2]))

