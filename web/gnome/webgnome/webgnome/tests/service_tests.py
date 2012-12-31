import datetime

from gnome.utilities.time_utils import round_time
from base import FunctionalTestBase


class ModelServiceTests(FunctionalTestBase):
    def create_model(self):
        return self.testapp.post('/model')

    def test_get_model_gets_a_valid_model(self):
        self.create_model()
        resp = self.testapp.get('/model', status=200)
        iso_rounded_now = round_time(datetime.datetime.now(), 3600).isoformat()

        self.assertEqual(resp.json_body['uncertain'], False)
        self.assertEqual(resp.json_body['start_time'], iso_rounded_now)
        self.assertEqual(resp.json_body['time_step'], 900.0)
        self.assertEqual(resp.json_body['duration_days'], 2)
        self.assertEqual(resp.json_body['duration_hours'], 0)
        self.assertEqual(resp.json_body['point_release_spills'], [])
        self.assertEqual(resp.json_body['wind_movers'], [])

    def test_get_model_returns_404_if_model_does_not_exist(self):
        resp = self.testapp.get('/model', status=404)
        self.assertEqual(resp.status_code, 404)

    def test_create_model_creates_a_model(self):
        resp = self.testapp.post('/model', status=200)

        self.assertTrue(resp.json_body['model_id'])
        self.assertEqual(resp.json_body['message'],  {
            'type': 'success',
            'text': 'Created a new model.'
        })

    def test_delete_model_deletes_model(self):
        self.create_model()
        resp = self.testapp.delete('/model', status=200)

        self.assertTrue(resp.json_body['model_id'])
        self.assertEqual(resp.json_body['message'],  {
            'type': 'success',
            'text': 'Deleted the current model.'
        })

        # Model doesn't exist anymore
        resp = self.testapp.get('/model', status=404)
        self.assertEqual(resp.status_code, 404)

    def test_create_model_creates_a_new_model_if_one_eixsts(self):
        self.create_model()

        # Get default settings
        resp = self.testapp.get('/model', status=200)
        default_uncertain = resp.json_body['uncertain']
        default_time_step = resp.json_body['time_step']

        # Change model settings and get a couple of the new values to test.
        self.test_post_settings()
        resp = self.testapp.get('/model', status=200)
        self.assertNotEqual(default_uncertain, resp.json_body['uncertain'])
        self.assertNotEqual(default_time_step, resp.json_body['time_step'])

        # Verify that the new model has default values.
        self.testapp.post('/model', status=200)
        resp = self.testapp.get('/model', status=200)
        self.assertEqual(resp.json_body['uncertain'], default_uncertain)
        self.assertEqual(resp.json_body['time_step'], default_time_step)

    def test_get_default_settings(self):
        self.create_model()
        resp = self.testapp.get('/model/settings', status=200)
        iso_rounded_now = round_time(datetime.datetime.now(), 3600).isoformat()

        self.assertEqual(resp.json_body['uncertain'], False)
        self.assertEqual(resp.json_body['start_time'], iso_rounded_now)
        self.assertEqual(resp.json_body['time_step'], 900.0)
        self.assertEqual(resp.json_body['duration_days'], 2)
        self.assertEqual(resp.json_body['duration_hours'], 0)

    def test_post_settings(self):
        self.create_model()
        start = datetime.datetime(2012, 12, 1, 2, 30)

        data = {
            'start_time': start.isoformat(),
            'uncertain': True,
            'time_step': 200,
            'duration_days': 20,
            'duration_hours': 1
        }

        resp = self.testapp.post_json('/model/settings', data)
        self.assertTrue(resp.json_body['success'])

        resp = self.testapp.get('/model/settings')

        self.assertEqual(resp.json_body['uncertain'], True)
        self.assertEqual(resp.json_body['start_time'], start.isoformat())
        self.assertEqual(resp.json_body['time_step'], 200.0)
        self.assertEqual(resp.json_body['duration_days'], 20)
        self.assertEqual(resp.json_body['duration_hours'], 1)


class WindMoverServiceTests(FunctionalTestBase):

    def get_safe_date(self, dt):
        """
        Return ``dt``, a :class:`datetime.datetime` value, stripped of micro-
        seconds and formatted with the `.isoformat()` function.

        This is to facilitate testing ``dt`` against a return value from
        model-related web services, will come back stripped of microseconds.

        Using .isoformat() prepares the object for JSON serialization.
        """
        return dt.replace(microsecond=0).isoformat()


    def create_model(self):
        return self.testapp.post('/model')

    def make_wind_mover_data(self, **kwargs):
        now = datetime.datetime.now()
        one_day = datetime.timedelta(days=1)
        dates = [now + one_day, now + one_day * 2, now + one_day * 3]

        timeseries = [
            {'datetime': dates[0], 'speed': 10, 'direction': 90},
            {'datetime': dates[1], 'speed': 20, 'direction': 180},
            {'datetime': dates[2], 'speed': 30, 'direction': 270},
        ]

        data = {
            'wind': {
                'timeseries': timeseries,
                'units': 'mps'
            },
            'is_active': True,
            'uncertain_duration': 4,
            'uncertain_time_delay': 2,
            'uncertain_speed_scale': 2,
            'uncertain_angle_scale': 3,
            'uncertain_angle_scale_units': 'deg'
        }

        if kwargs:
            data.update(**kwargs)

        data['wind']['timeseries'] = [
            dict(datetime=self.get_safe_date(val['datetime']),
                 speed=val['speed'], direction=val['direction'])
            for val in data['wind']['timeseries']]

        return data

    def test_wind_mover_create(self):
        self.create_model()
        data = self.make_wind_mover_data()

        resp = self.testapp.post_json('/mover/wind', data)
        mover_id = resp.json_body['id']

        self.assertTrue(resp.json_body['success'])
        self.assertTrue(mover_id)

        resp = self.testapp.get('/mover/wind/{0}'.format(mover_id))

        self.assertEqual(resp.json['is_active'], True)
        self.assertEqual(resp.json['wind']['units'], 'mps')

        winds = data['wind']['timeseries']
        self.assertEqual(resp.json['wind']['timeseries'], [
            {'direction': 90.0, 'speed': 10.0, 'datetime': winds[0]['datetime']},
            {'direction': 180.0, 'speed': 20.0, 'datetime': winds[1]['datetime']},
            {'direction': 270.0, 'speed': 30.0, 'datetime': winds[2]['datetime']}
        ])

        self.assertEqual(resp.json['uncertain_duration'], data['uncertain_duration'])
        self.assertEqual(resp.json['uncertain_time_delay'], data['uncertain_time_delay'])
        self.assertEqual(resp.json['uncertain_angle_scale'], data['uncertain_angle_scale'])
        self.assertEqual(resp.json['uncertain_angle_scale_units'], 'deg')

    def test_wind_mover_update(self):
        self.create_model()
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json('/mover/wind', data)
        mover_id = resp.json_body['id']

        safe_now = self.get_safe_date(datetime.datetime.now())
        data['wind']['timeseries'][0]['direction'] = 120
        data['wind']['timeseries'][0]['datetime'] = safe_now
        data['is_active'] = False
        data['uncertain_duration'] = 6

        resp = self.testapp.put_json('/mover/wind/{0}'.format(mover_id), data)
        self.assertEqual(resp.json['success'], True)

        resp = self.testapp.get('/mover/wind/{0}'.format(mover_id))

        self.assertEqual(resp.json['wind']['timeseries'][0]['direction'],
                         data['wind']['timeseries'][0]['direction'])
        self.assertEqual(resp.json['wind']['timeseries'][0]['datetime'],
                         safe_now)
        self.assertEqual(resp.json['is_active'], False)
        self.assertEqual(resp.json['uncertain_duration'], 6.0)

    def test_wind_mover_delete(self):
        self.create_model()
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json('/mover/wind', data)
        mover_id = resp.json_body['id']
        resp = self.testapp.delete_json('/mover/wind/{0}'.format(mover_id))
        self.assertEqual(resp.json['success'], True)
        self.testapp.get('/mover/wind/{0}'.format(mover_id), status=404)


class PointReleaseSpillServiceTests(FunctionalTestBase):
    def create_model(self):
        return self.testapp.post('/model')

    def make_spill_data(self, **kwargs):
        now = datetime.datetime.now()

        data = {
            'is_active': True,
            'release_time': now.isoformat(),
            'num_LEs': 900,
            'name': 'Point Release Spill',
            'start_position': [10, 100, 0],
            'windage': [1.2, 4.2],
            'persist': 900,
            'uncertain': False
        }

        if kwargs:
            data.update(**kwargs)

        return data

    def test_spill_create(self):
        self.create_model()
        data = self.make_spill_data()

        resp = self.testapp.post_json('/spill/point_release', data)
        spill_id = resp.json_body['id']

        self.assertTrue(resp.json_body['success'])
        self.assertTrue(spill_id)

        resp = self.testapp.get('/spill/point_release/{0}'.format(spill_id))

        self.assertEqual(resp.json['release_time'], data['release_time'])

    def test_spill_update(self):
        self.create_model()
        data = self.make_spill_data()
        resp = self.testapp.post_json('/spill/point_release', data)
        spill_id = resp.json_body['id']

        data['is_active'] = False
        data['release_Time'] = datetime.datetime.now().isoformat()

        resp = self.testapp.put_json('/spill/point_release/{0}'.format(spill_id), data)
        self.assertEqual(resp.json['success'], True)

        resp = self.testapp.get('/spill/point_release/{0}'.format(spill_id))
        self.assertEqual(resp.json['release_time'], data['release_time'])

    def test_spill_delete(self):
        self.create_model()
        data = self.make_spill_data()
        resp = self.testapp.post_json('/spill/point_release', data)
        spill_id = resp.json_body['id']
        resp = self.testapp.delete_json('/spill/point_release/{0}'.format(spill_id))
        self.assertEqual(resp.json['success'], True)
        self.testapp.get('/spill/point_release/{0}'.format(spill_id), status=404)
