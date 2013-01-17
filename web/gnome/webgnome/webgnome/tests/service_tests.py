import datetime

from gnome.utilities.time_utils import round_time
from base import FunctionalTestBase


class ModelHelperMixin(object):
    def create_model(self):
        resp = self.testapp.post('/model')
        self.base_url = str('/model/%s' % resp.json_body['id'])
        return resp

    def model_url(self, postfix):
        postfix = '/%s' % postfix if postfix[0] not in ('/', '?') else postfix
        return str('%s%s' % (self.base_url, postfix))


class ModelServiceTests(FunctionalTestBase, ModelHelperMixin):

    def test_get_model_gets_a_valid_model(self):
        self.create_model()
        resp = self.testapp.get(self.base_url, status=200)
        data = resp.json_body
        iso_rounded_now = round_time(datetime.datetime.now(), 3600).isoformat()

        self.assertEqual(data['is_uncertain'], False)
        self.assertEqual(data['start_time'], iso_rounded_now)
        self.assertEqual(data['time_step'], 0.25)
        self.assertEqual(data['duration_days'], 2)
        self.assertEqual(data['duration_hours'], 0)

        # We did not specify to include movers or spills.
        self.assertNotIn('surface_release_spills', data)
        self.assertNotIn('wind_movers', data)

    def test_get_model_includes_movers_if_requested(self):
        self.create_model()
        resp = self.testapp.get(self.model_url('?include_movers=1'), status=200)
        data = resp.json_body

        self.assertEqual(data['wind_movers'], [])

    def test_get_model_includes_spills_if_requested(self):
        self.create_model()
        resp = self.testapp.get(self.model_url('?include_spills=1'), status=200)
        data = resp.json_body

        self.assertEqual(data['surface_release_spills'], [])

    def test_get_model_returns_404_if_model_does_not_exist(self):
        resp = self.testapp.get('/model/234343', status=404)
        self.assertEqual(resp.status_code, 404)

    def test_create_model_creates_a_model(self):
        resp = self.create_model()
        self.assertTrue(resp.json_body['id'])
        self.assertEqual(resp.json_body['message'],  {
            'type': 'success',
            'text': 'Created a new model.'
        })

    def test_delete_model_deletes_model(self):
        self.create_model()
        resp = self.testapp.delete(self.base_url, status=200)

        # Model doesn't exist anymore
        resp = self.testapp.get(self.base_url, status=404)
        self.assertEqual(resp.status_code, 404)

    def test_put_settings(self):
        self.create_model()
        start = datetime.datetime(2012, 12, 1, 2, 30)

        data = {
            'start_time': start.isoformat(),
            'is_uncertain': True,
            'time_step': 200,
            'duration_days': 20,
            'duration_hours': 1
        }

        resp = self.testapp.put_json(self.base_url, data)
        self.assertTrue(resp.json_body['id'])

        resp = self.testapp.get(self.base_url)

        self.assertEqual(resp.json_body['is_uncertain'], True)
        self.assertEqual(resp.json_body['start_time'], start.isoformat())
        self.assertEqual(resp.json_body['time_step'], 200.0)
        self.assertEqual(resp.json_body['duration_days'], 20)
        self.assertEqual(resp.json_body['duration_hours'], 1)


class ModelRunnerServiceTests(FunctionalTestBase, ModelHelperMixin):

    def test_get_first_step(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        self.testapp.get('/long_island')

        # Post to runner URL to get the first step.
        resp = self.testapp.post_json(self.model_url('runner'))
        data = resp.json_body
        self.assertIn('foreground_00000.png', data['time_step']['url'])
        self.assertEqual(data['time_step']['id'], 0)

    def test_get_additional_steps(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        self.testapp.get('/long_island')

        # Post to runner URL to get the first step and expected # of time steps.
        url = self.model_url('runner')
        resp = self.testapp.post_json(url)
        num_steps = len(resp.json_body['expected_time_steps'])

        for step in range(num_steps):
            # Skip the first step because we received it in the POST.
            if step == 0:
                continue
            resp = self.testapp.get(url)
            data = resp.json_body
            # TODO: Add more value tests here.
            self.assertEqual(data['time_step']['id'], step)

        # The model should now be exhausted and return a 404 if the user tries
        # to run it.
        resp = self.testapp.get(url, status=404)
        self.assertEqual(resp.status_code, 404)

    def test_restart_runner_after_finished(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        self.testapp.get('/long_island')

        # Post to runner URL to get the first step and expected # of time steps.
        url = self.model_url('runner')
        resp = self.testapp.post_json(url)
        num_steps = len(resp.json_body['expected_time_steps'])

        for step in range(num_steps):
            # Skip the first step because we received it in the POST.
            if step == 0:
                continue
            self.testapp.get(url)

        resp = self.testapp.get(url, status=404)
        self.assertEqual(resp.status_code, 404)

        # Restart the model runner by POSTing to its URL.
        resp = self.testapp.post_json(url)

        # Verify that we can now get steps
        self.assertIn('foreground_00000.png', resp.json_body['time_step']['url'])
        resp = self.testapp.get(url)
        self.assertIn('foreground_00001.png', resp.json_body['time_step']['url'])


class WindMoverServiceTests(FunctionalTestBase, ModelHelperMixin):

    def setUp(self):
        super(WindMoverServiceTests, self).setUp()
        self.create_model()
        self.collection_url = self.model_url('/mover/wind')

    def get_mover_url(self, mover_id):
        return str('%s/%s' % (self.collection_url, mover_id))

    def get_safe_date(self, dt):
        """
        Return ``dt``, a :class:`datetime.datetime` value, stripped of micro-
        seconds and formatted with the `.isoformat()` function.

        This is to facilitate testing ``dt`` against a return value from
        model-related web services, will come back stripped of microseconds.

        Using .isoformat() prepares the object for JSON serialization.
        """
        return dt.replace(microsecond=0).isoformat()

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
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json(self.collection_url, data)
        mover_id = resp.json_body['id']

        self.assertTrue(mover_id)

        resp = self.testapp.get(self.get_mover_url(mover_id))

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
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json(self.collection_url, data)
        mover_id = resp.json_body['id']

        safe_now = self.get_safe_date(datetime.datetime.now())
        data['wind']['timeseries'][0]['direction'] = 120
        data['wind']['timeseries'][0]['datetime'] = safe_now
        data['uncertain_duration'] = 6

        resp = self.testapp.put_json(self.get_mover_url(mover_id), data)
        self.assertTrue(resp.json_body['id'])

        resp = self.testapp.get(self.get_mover_url(mover_id))

        self.assertEqual(resp.json['wind']['timeseries'][0]['direction'],
                         data['wind']['timeseries'][0]['direction'])
        self.assertEqual(resp.json['wind']['timeseries'][0]['datetime'],
                         safe_now)
        self.assertEqual(resp.json['uncertain_duration'], 6.0)

    def test_wind_mover_delete(self):
        data = self.make_wind_mover_data()

        # create a mover
        resp = self.testapp.post_json(self.collection_url, data)
        mover_id = resp.json_body['id']

        # delete the mover
        mover_url = self.get_mover_url(mover_id)
        resp = self.testapp.delete_json(mover_url)
        self.testapp.get(mover_url, status=404)


class SurfaceReleaseSpillServiceTests(FunctionalTestBase, ModelHelperMixin):
    def setUp(self):
        super(SurfaceReleaseSpillServiceTests, self).setUp()
        self.create_model()
        self.collection_url = self.model_url('/spill/surface_release')

    def get_spill_url(self, spill_id):
        return str('%s/%s' % (self.collection_url, spill_id))

    def make_spill_data(self, **kwargs):
        now = datetime.datetime.now()

        data = {
            'is_active': True,
            'release_time': now.isoformat(),
            'num_elements': 900,
            'name': 'Point Release Spill',
            'start_position': [10, 100, 0],
            'windage_range': [1.2, 4.2],
            'windage_persist': 900,
            'is_uncertain': False
        }

        if kwargs:
            data.update(**kwargs)

        return data

    def test_spill_create(self):
        data = self.make_spill_data()
        resp = self.testapp.post_json(self.collection_url, data)
        spill_id = resp.json_body['id']

        self.assertTrue(spill_id)

        resp = self.testapp.get(self.get_spill_url(spill_id))

        self.assertEqual(resp.json['release_time'], data['release_time'])

    def test_spill_update(self):
        data = self.make_spill_data()
        resp = self.testapp.post_json(self.collection_url, data)
        spill_id = resp.json_body['id']

        data['is_active'] = False
        data['release_Time'] = datetime.datetime.now().isoformat()

        spill_url = self.get_spill_url(spill_id)

        resp = self.testapp.put_json(spill_url, data)
        self.assertTrue(resp.json['id'])

        resp = self.testapp.get(spill_url)
        self.assertEqual(resp.json['release_time'], data['release_time'])

    def test_spill_delete(self):
        data = self.make_spill_data()
        resp = self.testapp.post_json(self.collection_url, data)
        spill_id = resp.json_body['id']
        spill_url = self.get_spill_url(spill_id)
        resp = self.testapp.delete_json(spill_url)
        self.testapp.get(spill_url, status=404)


class MapServiceTests(FunctionalTestBase, ModelHelperMixin):
    def setUp(self):
        super(MapServiceTests, self).setUp()
        self.create_model()
        self.url = self.model_url('map')

    def test_create_map(self):
        data = {
            'filename': '/data/lakeerie.bna',
            'name': 'Lake Eerie',
            'refloat_halflife': 6 * 3600
        }
        resp = self.testapp.post_json(self.url, data)
        resp_data = resp.json_body

        self.assertIn(data['filename'], resp_data['filename'])
        self.assertEqual(data['name'], resp_data['name'])
        self.assertEqual(data['refloat_halflife'], resp_data['refloat_halflife'])

    def test_get_map(self):
        data = {
            'filename': '/data/lakeerie.bna',
            'name': 'Lake Eerie',
            'refloat_halflife': 6 * 3600
        }
        resp = self.testapp.post_json(self.url, data)
        self.assertTrue(resp.json_body['id'])

        resp = self.testapp.get(self.url)
        resp_data = resp.json_body
        self.assertIn(data['filename'], resp_data['filename'])
        self.assertEqual(data['name'], resp_data['name'])
        self.assertEqual(data['refloat_halflife'], resp_data['refloat_halflife'])
