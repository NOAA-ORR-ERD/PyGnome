import datetime
import os
import shutil

from time import gmtime
from gnome.utilities.time_utils import round_time
from base import FunctionalTestBase
from webgnome import util


class ModelHelperMixin(object):
    def create_model(self):
        resp = self.testapp.post('/model')
        self.model_id = resp.json_body['id']
        self.base_url = str('/model/%s' % self.model_id)
        return resp

    def model_url(self, postfix):
        """
        Return a URL specific to the running model. The URL will include the
        model's base URL (/model/{id}) followed by ``postfix``.
        """
        postfix = '/%s' % postfix if postfix[0] not in ('/', '?') else postfix
        return str('%s%s' % (self.base_url, postfix))


class ModelServiceTests(FunctionalTestBase, ModelHelperMixin):

    def test_get_model_gets_a_valid_model(self):
        self.create_model()
        resp = self.testapp.get(self.base_url, status=200)
        data = resp.json_body
        iso_rounded_now = round_time(datetime.datetime.now(), 3600).isoformat()

        self.assertEqual(data['uncertain'], False)
        self.assertEqual(data['start_time'], iso_rounded_now)
        self.assertEqual(data['time_step'], 0.25)
        self.assertEqual(data['duration_days'], 1)
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
            'uncertain': True,
            'time_step': 200,
            'duration_days': 20,
            'duration_hours': 1
        }

        resp = self.testapp.put_json(self.base_url, data)
        self.assertTrue(resp.json_body['id'])

        resp = self.testapp.get(self.base_url)

        self.assertEqual(resp.json_body['uncertain'], True)
        self.assertEqual(resp.json_body['start_time'], start.isoformat())
        self.assertEqual(resp.json_body['time_step'], 200.0)
        self.assertEqual(resp.json_body['duration_days'], 20)
        self.assertEqual(resp.json_body['duration_hours'], 1)


class GnomeRunnerServiceTests(FunctionalTestBase, ModelHelperMixin):

    def test_get_first_step(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        location_url = self.model_url('/location_file/long_island')
        resp = self.testapp.get(location_url)
        self.testapp.put_json(self.base_url, resp.json_body)

        # Post to runner URL to get the first step.
        resp = self.testapp.post_json(self.model_url('runner'))
        data = resp.json_body
        self.assertIn('foreground_00000.png', data['time_step']['url'])
        self.assertEqual(data['time_step']['id'], 0)

    def test_get_additional_steps(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        location_url = self.model_url('/location_file/long_island')
        resp = self.testapp.get(location_url)
        self.testapp.put_json(self.base_url, resp.json_body)

        # Post to runner URL to get the first step and expected # of time steps.
        runner_url = self.model_url('runner')
        resp = self.testapp.post_json(runner_url)
        num_steps = len(resp.json_body['expected_time_steps'])

        for step in range(num_steps):
            # Skip the first step because we received it in the POST.
            if step == 0:
                continue
            resp = self.testapp.get(runner_url)
            data = resp.json_body
            # TODO: Add more value tests here.
            self.assertEqual(data['time_step']['id'], step)

        # The model should now be exhausted and return a 404 if the user tries
        # to run it.
        resp = self.testapp.get(runner_url, status=404)
        self.assertEqual(resp.status_code, 404)

    def test_restart_runner_after_finished(self):
        self.create_model()

        # Load the Long Island script parameters into the model.
        location_url = self.model_url('/location_file/long_island')
        resp = self.testapp.get(location_url)
        self.testapp.put_json(self.base_url, resp.json_body)

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
            [dates[0], 10, 30],
            [dates[1], 20, 180],
            [dates[2], 30, 270]
        ]

        data = {
            'wind': {
                'timeseries': timeseries,
                'units': 'mps'
            },
            'uncertain_duration': 4,
            'uncertain_time_delay': 2,
            'uncertain_speed_scale': 2,
            'uncertain_angle_scale': 3,
            'uncertain_angle_scale_units': 'deg'
        }

        if kwargs:
            data.update(**kwargs)

        data['wind']['timeseries'] = [
            [self.get_safe_date(val[0]), val[1], val[2]]
            for val in data['wind']['timeseries']]

        return data

    def jtest_wind_mover_create(self):
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json(self.collection_url, data)
        mover_id = resp.json_body['id']

        self.assertTrue(mover_id)

        resp = self.testapp.get(self.get_mover_url(mover_id))

        self.assertEqual(resp.json['on'], True)
        self.assertEqual(resp.json['wind']['units'], 'mps')

        winds = data['wind']['timeseries']
        self.assertEqual(resp.json['wind']['timeseries'], [
            [winds[0]['datetime'], 10, 90],
            [winds[1]['datetime'], 20, 180],
            [winds[2]['datetime'], 30, 270]
        ])

        self.assertEqual(resp.json['uncertain_duration'], data['uncertain_duration'])
        self.assertEqual(resp.json['uncertain_time_delay'], data['uncertain_time_delay'])
        self.assertEqual(resp.json['uncertain_angle_scale'], data['uncertain_angle_scale'])
        self.assertEqual(resp.json['uncertain_angle_scale_units'], 'deg')
        self.assertEqual(resp.json['active_start'],
                         datetime.datetime(*gmtime(0)[:6]).isoformat())
        self.assertEqual(resp.json['active_stop'],
                         datetime.datetime(2038,1,18,0,0,0).isoformat())

    def test_wind_mover_update(self):
        data = self.make_wind_mover_data()
        resp = self.testapp.post_json(self.collection_url, data)
        mover_id = resp.json_body['id']

        safe_now = self.get_safe_date(datetime.datetime.now())
        data['wind']['timeseries'][0][0] = safe_now
        data['wind']['timeseries'][0][2] = 120
        data['uncertain_duration'] = 6

        resp = self.testapp.put_json(self.get_mover_url(mover_id), data)
        self.assertTrue(resp.json_body['id'])

        resp = self.testapp.get(self.get_mover_url(mover_id))

        self.assertEqual(resp.json['wind']['timeseries'][0][2],
                         data['wind']['timeseries'][0][2])
        self.assertEqual(resp.json['wind']['timeseries'][0][0], safe_now)
        self.assertEqual(resp.json['uncertain_duration'], 6.0)

    def test_wind_mover_update_is_active_fields(self):
        active_start = datetime.datetime.now().isoformat()
        active_stop = datetime.datetime.now().isoformat()

        data = self.make_wind_mover_data()
        data['active_start'] = active_start
        data['active_stop'] = active_stop
        resp = self.testapp.post_json(self.collection_url, data)

        self.assertEqual(resp.json['active_start'], active_start)
        self.assertEqual(resp.json['active_stop'], active_stop)

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
            'uncertain': False
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
        data['release_time'] = datetime.datetime.now().isoformat()

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

    def copy_map(self, location_file, map_name, destination_name=None):
        """
        Copy a map with the filename ``map_name`` from the location file
        ``location_file`` to the current model's data directory.

        If ``destination_name`` is None, use ``map_name`` as the new filename.
        """
        destination_name = destination_name or map_name
        original_file = os.path.join(self.project_root, 'location_files',
                                     location_file, 'data', map_name)
        destination_file = os.path.join(self.project_root, 'static',
                                        self.settings['model_data_dir'],
                                        self.model_id, 'data', destination_name)
        shutil.copy(original_file, destination_file)

        return destination_name

    def test_create_map(self):
        filename = self.copy_map('long_island', 'LongIslandSoundMap.BNA')
        data = {
            'filename': filename,
            'name': 'Long Island',
            'refloat_halflife': 6 * 3600
        }
        resp = self.testapp.post_json(self.url, data)
        resp_data = resp.json_body

        self.assertEqual(data['name'], resp_data['name'])
        self.assertEqual(data['refloat_halflife'], resp_data['refloat_halflife'])
        self.assertEqual(resp_data['map_bounds'],
                         [[-73.083328, 40.922832], [-73.083328, 41.330833],
                          [-72.336334, 41.330833], [-72.336334, 40.922832]])

    def test_update_map(self):
        filename = self.copy_map('long_island', 'LongIslandSoundMap.BNA')
        data = {
            'filename': filename,
            'name': 'Long Island',
            'refloat_halflife': 6 * 3600
        }
        self.testapp.post_json(self.url, data)

        data['name'] = 'Long Island 2'
        data['refloat_halflife'] = 10

        resp = self.testapp.put_json(self.url, data)
        resp_data = resp.json_body

        self.assertEqual(data['name'], resp_data['name'])
        self.assertEqual(data['refloat_halflife'],
                         resp_data['refloat_halflife'])

    def test_get_map(self):
        filename = self.copy_map('long_island', 'LongIslandSoundMap.BNA')
        data = {
            'filename': filename,
            'name': 'Long Island',
            'refloat_halflife': 6 * 3600
        }
        self.testapp.post_json(self.url, data)

        resp = self.testapp.get(self.url)
        resp_data = resp.json_body

        self.assertEqual(data['name'], resp_data['name'])
        self.assertEqual(data['refloat_halflife'], resp_data['refloat_halflife'])
        self.assertEqual(resp_data['map_bounds'],
                         [[-73.083328, 40.922832], [-73.083328, 41.330833],
                          [-72.336334, 41.330833], [-72.336334, 40.922832]])


class CustomMapServiceTests(FunctionalTestBase, ModelHelperMixin):
    def setUp(self):
        super(CustomMapServiceTests, self).setUp()
        self.create_model()
        self.url = self.model_url('custom_map')

    def test_post(self):
        data = {
            'north_lat': 46.298,
            'east_lon': -123.891,
            'west_lon': -124.092,
            'south_lat': 46.186,
            'resolution': 'c',
            'refloat_halflife': 2,
            'name': 'BNA Map'
        }

        resp = self.testapp.post_json(self.url, data)

        self.assertEqual(resp.json_body['map_bounds'],
                         [[-124.092, 46.186], [-124.092, 46.298],
                          [-123.891, 46.298], [-123.891, 46.186]])
        self.assertEqual(resp.json_body['refloat_halflife'], 2.0)
        self.assertEqual(resp.json_body['name'], 'BNA Map')

    def test_post_with_form_errors(self):
        data = {
            "north_lat": "45.829",
            "east_lon": "-123.398",
            "south_lat": "43.069",
            "west_lon": "-126.914",
            'resolution': 'c',
            'refloat_halflife': 2,
            'name': 'BNA Map'
        }

        resp = self.testapp.post_json(self.url, data, status=500)
        body = resp.json_body

        self.assertEqual(body['status'], 'error')
        self.assertEqual(body['errors'][0]['name'], 'map')
        self.assertEqual(body['errors'][0]['description'],
                         'No shoreline segments found in this domain')


class LocationFileServiceTests(FunctionalTestBase, ModelHelperMixin):
    long_island = {
        u'wind_movers': [
            {
                u'on': True, u'name': u'Wind Mover',
                u'uncertain_angle_scale': 0.4,
                u'uncertain_duration': 10800.0,
                u'active_start': u'1970-01-01T00:00:00',
                u'active_stop': u'2038-01-18T00:00:00',
                u'uncertain_angle_scale_units': u'rad',
                u'uncertain_time_delay': 0.0,
                u'wind': {
                    u'units': u'mps', u'description': None,
                    u'source_type': u'manual', u'updated_at': None,
                    u'longitude': None, u'source_id': None,
                    u'timeseries': [[u'2013-02-05T17:00:00', 30.0, 50.0],
                                    [u'2013-02-06T11:00:00', 30.0, 50.0],
                                    [u'2013-02-06T23:00:00',
                                     20.000000000000004, 25.0],
                                    [u'2013-02-07T11:00:00', 25.0, 10.0],
                                    [u'2013-02-07T23:00:00', 25.0, 180.0]],
                    u'latitude': None,
                },
                u'uncertain_speed_scale': 2.0
            }
        ],
        u'map': {
            u'map_bounds': [[-73.083328, 40.922832],
                            [-73.083328, 41.330833],
                            [-72.336334, 41.330833],
                            [-72.336334, 40.922832]],
            u'name': u'Long Island Sound', u'refloat_halflife': 21600.0,
            u'relative_path': u'location_files/long_island/data/LongIslandSoundMap.BNA'
        },
        u'start_time': u'2013-02-05T17:00:00',
        u'random_movers': [
            {
                u'diffusion_coef': 500000.0,
                u'name': u'Random Mover',
                u'on': True,
                u'active_start': u'1970-01-01T00:00:00',
                u'active_stop': u'2038-01-18T00:00:00',
            }
        ],
        u'uncertain': False,
        u'surface_release_spills': [
            {
                u'end_position': [-72.419992, 41.20212, 0.0],
                u'end_release_time': u'2013-02-05T17:00:00',
                u'is_active': True,
                u'windage_range': [0.01, 0.04],
                u'num_elements': 1000,
                u'name': u'Long Island Spill',
                u'start_position': [-72.419992, 41.20212, 0.0],
                u'release_time': u'2013-02-05T17:00:00',
                u'windage_persist': 900.0,
            }
        ],
        u'time_step': 0.25, u'duration_hours': 0,
        u'duration_days': 1,
    }

    def setUp(self):
        super(LocationFileServiceTests, self).setUp()
        self.create_model()
        self.url = self.model_url('location_file')
        self.maxDiff = 5000

    def test_post_to_long_island_updates_the_model_configuration(self):
        url = self.model_url('/location_file/long_island')
        resp = self.testapp.get(url)
        resp = self.testapp.put_json(self.base_url, resp.json_body)
        body = resp.json_body
        util.delete_keys_from_dict(body, [u'id'])
        self.assertEqual(body, self.long_island)


class LocationFileWizardServiceTests(FunctionalTestBase, ModelHelperMixin):
    def setUp(self):
        super(LocationFileWizardServiceTests, self).setUp()
        self.create_model()

    def test_get_wizard_returns_html_fragment(self):
        wizard_url = self.model_url('/location_file/test/wizard')
        resp = self.testapp.get(wizard_url)
        data = resp.json_body

        self.assertIn("""<form class='wizard form page hide' id="test_wizard" title="Test Location File">""",
                      data['html'])
        self.assertIn('<div class="step hidden" data-reference-form=model-settings>',
                      data['html'])
        self.assertIn('<select class="type input-small" data-value="wizard.custom_stuff" id="custom_stuff" name="custom_stuff">',
                      data['html'])

    def test_post_to_test_wizard_changes_model(self):
        resp = self.testapp.get(self.base_url)
        model_data = resp.json_body
        self.assertEqual(model_data['duration_days'], 1)

        wizard_url = self.model_url('/location_file/test/wizard')
        resp = self.testapp.put_json(wizard_url, {'use_custom_thing': 'yes'})

        resp = self.testapp.get(self.base_url)
        model_data = resp.json_body
        self.assertEqual(model_data['duration_days'], 10)
