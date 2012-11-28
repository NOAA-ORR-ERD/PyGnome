import json
import numpy

from collections import OrderedDict
from webob.multidict import MultiDict

from base import FunctionalTestBase, UnitTestBase
from webgnome.model_manager import ModelManager
from webgnome.views.movers import create_wind_mover, update_wind_mover


class WindMoverFixtures(object):
    def create_wind_mover_data(self, **kwargs):
        data = OrderedDict([
            ('is_active', 'y'),
            ('start_time', '0'),
            ('duration', '3'),
            ('speed_scale', '2'),
            ('total_angle_scale', '0.4'),
            ('total_angle_scale_type', 'rad')
        ])

        data.update(**kwargs)
        return data

    def create_time_series_data(self, num_time_series=1, **kwargs):
        time_series_data = []

        for index in xrange(num_time_series):
            prefix = 'timeseries-%s-' % index

            time_series = {
                'date': '11/20/2012',
                'hour': index,
                'minute': '0',
                'auto_increment_time_by': '1',
                'direction': 'S',
                'direction_degrees': '',
                'speed': '10',
                'speed_type': 'knots'
            }

            time_series.update(**kwargs)

            # Use special ``wtforms.core.FieldList`` keys.
            for key, val in time_series.items():
                time_series[prefix + key] = time_series.pop(key)

            time_series_data.append(time_series)

        return time_series_data


class WindMoverFunctionalTests(FunctionalTestBase, WindMoverFixtures):

    def create_model(self):
        return self.testapp.post('/model/create', OrderedDict([
            ('confirm_new', True)
        ]))

    def test_create_wind_mover_with_multiple_time_series(self):
        data = self.create_wind_mover_data()

        for item in self.create_time_series_data(5):
            data.update(item)

        self.create_model()
        resp = self.testapp.post('/model/mover/wind', data)

        json_resp = json.loads(resp.body)

        self.assertEqual(json_resp['form_html'], None)
        self.assertEqual(json_resp['type'], 'mover')
        self.assertIsNotNone(json_resp['id'])

    def test_create_wind_mover_with_one_time_series(self):
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1)[0])

        self.create_model()
        resp = self.testapp.post('/model/mover/wind', data)

        json_resp = json.loads(resp.body)

        self.assertEqual(json_resp['form_html'], None)
        self.assertEqual(json_resp['type'], 'mover')
        self.assertIsNotNone(json_resp['id'])


class WindMoverUnitTests(UnitTestBase, WindMoverFixtures):
    def get_model_request(self):
        """
        Get a :class:`pyramid.testing.DummyRequest` object with a model ID
        in its session that points to running :class:`gnome.model.Model`.
        """
        request = self.get_request()
        request.registry.settings.Model = ModelManager()
        model = request.registry.settings.Model.create()
        request.registry.settings['model_session_key'] = 'model_key'
        request.session['model_key'] = model.id
        return model, request

    def test_create_wind_mover_single_time_series(self):
        self.config.add_route('create_wind_mover', '/mover')

        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1)[0])

        request.POST = MultiDict(data)
        resp = create_wind_mover(request)

        mover = model.get_mover(resp['id'])

        self.assertEqual(len(mover.timeseries), 1)

        self.assertEqual(
            mover.timeseries[0].date,
            numpy.datetime64('2012-11-20 00:00:00.000001').astype('object'))

        self.assertEqual(mover.timeseries[0].direction, 'Degrees true')
        self.assertEqual(mover.timeseries[0].direction_degrees, 180.0)
        self.assertEqual(mover.timeseries[0].speed, 10.0)

    def test_create_wind_mover_multiple_time_series(self):
        self.config.add_route('create_wind_mover', '/mover')

        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()

        for item in self.create_time_series_data(3):
            data.update(item)

        request.POST = MultiDict(data)
        resp = create_wind_mover(request)

        mover = model.get_mover(resp['id'])
        self.assertEqual(len(mover.timeseries), 3)

        for idx, item in enumerate(mover.timeseries):
            self.assertEqual(
                mover.timeseries[idx].date,
                numpy.datetime64(
                    '2012-11-20 0%s:00:00.000001' % idx).astype('object'))

        # The following values are correct, so why are they flipped?
        self.assertEqual(mover.timeseries[0].speed, 10.0)
        self.assertEqual(mover.timeseries[0].direction, 'Degrees true')
        self.assertEqual(mover.timeseries[0].direction_degrees, 180.0)

    def test_update_wind_mover_single_time_series(self):
        self.config.add_route('create_wind_mover', '/mover')
        self.config.add_route('update_wind_mover', '/mover/{id}')

        # Create the wind mover
        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1)[0])
        request.POST = MultiDict(data)
        resp = create_wind_mover(request)
        mover = model.get_mover(resp['id'])

        # The update wind mover form should have timeseries data.
        request.method = 'GET'
        request.POST = MultiDict()
        request.matchdict = MultiDict({'id': resp['id']})
        resp = update_wind_mover(request)

        # Verify that the update form has the timeseries data we submitted
        # plus an additional "Add" form with the default values.
        self.assertTrue(resp['form_html'].find(
            '<input class="direction_degrees" id="" '
            'name="timeseries-0-direction_degrees" '
            'type="text" value="180.0">') > 1)

        self.assertTrue(resp['form_html'].find(
            '<input class="direction_degrees" id="" '
            'name="timeseries-1-direction_degrees" type="text" value="">') > 1)

        # Update the wind mover with a new time series
        data = self.create_wind_mover_data()
        for item in self.create_time_series_data(1, speed=100):
            data.update(item)
        request.method = 'POST'
        request.POST = MultiDict(data)
        request.matchdict = MultiDict({'id': mover.id})
        resp = update_wind_mover(request)

        self.assertEqual(resp['message']['type'], 'success')
        self.assertEqual(resp['id'], mover.id)

        mover = model.get_mover(mover.id)
        for wind in mover.timeseries:
            self.assertEqual(wind.speed, 100)

    def test_update_wind_mover_single_time_series(self):
        self.config.add_route('create_wind_mover', '/mover')
        self.config.add_route('update_wind_mover', '/mover/{id}')

        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1)[0])

        request.POST = MultiDict(data)
        resp = create_wind_mover(request)
        self.assertTrue(resp['id'])
        self.assertEqual(resp['form_html'], None)

        mover = model.get_mover(resp['id'])
        self.assertEqual(len(mover.timeseries), 1)

        # The update wind mover form should have timeseries data.
        request.method = 'GET'
        request.POST = MultiDict()
        request.matchdict = MultiDict({'id': resp['id']})
        resp = update_wind_mover(request)

        self.assertTrue(resp['form_html'].find(
            '<input class="direction_degrees" id="" '
            'name="timeseries-0-direction_degrees" type="text" value="180.0">') > 1)

        data = self.create_wind_mover_data()
        for item in self.create_time_series_data(1, speed=100):
            data.update(item)

        request.method = 'POST'
        request.POST = MultiDict(data)
        request.matchdict = MultiDict({'id': mover.id})
        resp = update_wind_mover(request)

        self.assertEqual(resp['message']['type'], 'success')
        self.assertEqual(resp['id'], mover.id)

        mover = model.get_mover(mover.id)
        self.assertEqual(len(mover.timeseries), 1)
        for wind in mover.timeseries:
            self.assertEqual(wind.speed, 100)

    def test_update_wind_mover_with_field_error(self):
        self.config.add_route('create_wind_mover', '/mover')
        self.config.add_route('update_wind_mover', '/mover/{id}')

        # Create the wind mover
        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1)[0])
        request.POST = MultiDict(data)
        resp = create_wind_mover(request)
        mover = model.get_mover(resp['id'])

        # Verify that the update form has the timeseries data we submitted
        # plus an additional "Add" form with the default values.
        request.method = 'GET'
        request.POST = MultiDict()
        request.matchdict = MultiDict({'id': resp['id']})
        resp = update_wind_mover(request)

        self.assertTrue(resp['form_html'].find(
            '<input class="direction_degrees" id="" '
            'name="timeseries-0-direction_degrees" '
            'type="text" value="180.0">') > 1)

        self.assertTrue(resp['form_html'].find(
            '<input class="direction_degrees" id="" '
            'name="timeseries-1-direction_degrees" type="text" value="">') > 1)

        # Update the wind mover with a new time series
        data = self.create_wind_mover_data()
        for item in self.create_time_series_data(3, speed=100):
            data.update(item)
        request.method = 'POST'
        request.POST = MultiDict(data)
        request.matchdict = MultiDict({'id': mover.id})
        resp = update_wind_mover(request)

        self.assertEqual(resp['message']['type'], 'success')
        self.assertEqual(resp['id'], mover.id)

        mover = model.get_mover(mover.id)
        self.assertEqual(len(mover.timeseries), 3)
        for wind in mover.timeseries:
            self.assertEqual(wind.speed, 100)
