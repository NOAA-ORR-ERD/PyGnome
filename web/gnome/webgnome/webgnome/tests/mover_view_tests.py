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

        time_series = mover.timeseries
        self.assertEqual(len(time_series), 1)

        self.assertEqual(
            time_series[0][0],
            numpy.datetime64('2012-11-20 00:00:00.000001'))

        self.assertEqual(time_series[0][1][0], 180.0)
        self.assertEqual(time_series[0][1][1], 10.0)

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

        time_series = mover.timeseries
        self.assertEqual(len(time_series), 3)

        for idx, item in enumerate(time_series):
            self.assertEqual(
                time_series[idx][0],
                numpy.datetime64('2012-11-20 0%s:00:00.000001' % idx))

        self.assertEqual(time_series[0][1][0], 180.0)
        self.assertEqual(time_series[0][1][1], 10.0)

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

        request.method = 'GET'
        request.POST = MultiDict()
        request.matchdict = MultiDict({'id': resp['id']})
        resp = update_wind_mover(request)
        print resp['form_html']
