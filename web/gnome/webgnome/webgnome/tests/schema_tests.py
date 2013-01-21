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

        data = {
            'datetime': dt,
            'speed': 10.5,
            'direction': 180.5
        }

        wind_value = schema.WindValueSchema().serialize(data)

        self.assertEqual(wind_value['direction'], str(data['direction']))
        self.assertEqual(wind_value['speed'], str(data['speed']))
        self.assertEqual(wind_value['datetime'], dt_without_tz.isoformat())

    def test_deserialize_strips_timezone(self):
        dt = datetime.datetime.now(timezone('US/Pacific'))
        dt_without_tz = dt.replace(tzinfo=None)

        data = {
            'datetime': dt.isoformat(),
            'speed': '10.5',
            'direction': '180.5'
        }

        wind_value = schema.WindValueSchema().deserialize(data)

        self.assertEqual(wind_value['direction'], float(data['direction']))
        self.assertEqual(wind_value['speed'], float(data['speed']))
        self.assertEqual(wind_value['datetime'], dt_without_tz)

    def test_speed_must_be_nonzero(self):
        data = {
            'datetime': datetime.datetime.now(),
            'speed': 0,
            'direction': 180.5
        }

        self.assertRaises(colander.Invalid,
                          schema.WindValueSchema().deserialize, data)


class WindSchemaTests(TestCase):

    def get_test_data(self):
        timeseries = [
            {
                'datetime': datetime.datetime(year=2013, month=1, day=1,
                                              hour=12, minute=30,
                                              tzinfo=timezone('US/Pacific')),
                'speed': 20.5,
                'direction': 300.5
            },
            {
                'datetime': datetime.datetime(year=2013, month=2, day=1,
                                              hour=1, minute=30,
                                              tzinfo=timezone('US/Pacific')),
                'speed': 20.5,
                'direction': 300.5
            },
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
            self.assertEqual(wind_value['direction'], str(test_val['direction']))
            self.assertEqual(wind_value['speed'], str(test_val['speed']))
            self.assertEqual(wind_value['datetime'],
                             test_val['datetime'].replace(
                                 tzinfo=None).isoformat())

    def test_deserialize(self):
        data = self.get_test_data()

        for idx, wind_value in enumerate(data['timeseries']):
            # Save a timezone-unaware version so we can more easily test it as
            # the expected result after deserializing.
            data['timeseries'][idx]['expected_datetime'] = \
                str(wind_value['datetime'].replace(tzinfo=None))
            data['timeseries'][idx]['datetime'] = str(wind_value['datetime'])
            data['timeseries'][idx]['speed'] = str(wind_value['speed'])
            data['timeseries'][idx]['direction'] = str(wind_value['direction'])

        wind = schema.WindSchema().deserialize(data)

        for idx, wind_value in enumerate(wind['timeseries']):
            test_values= data['timeseries'][idx]
            self.assertEqual(wind_value[1][0], float(test_values['speed']))
            self.assertEqual(wind_value[1][1], float(test_values['direction']))
            self.assertEqual(wind_value[0],
                             numpy.datetime64(test_values['expected_datetime']))
