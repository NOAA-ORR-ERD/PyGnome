import json

from collections import OrderedDict
from webob.multidict import MultiDict

from base import FunctionalTestBase, UnitTestBase
from webgnome.model_manager import ModelManager
from webgnome.views.movers import create_wind_mover
from webgnome import util


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

    def create_time_series_data(self, index, **kwargs):
        prefix = 'time_series-%s-' % index

        data = {
            'date': '11/20/2012',
            'hour': '0',
            'minute': '0',
            'auto_increment_time_by': '1',
            'direction': 'S',
            'direction_degrees': '',
            'speed': '10',
            'speed_type': 'knots'
        }

        data.update(**kwargs)

        # Use special ``wtforms.core.FieldList`` keys.
        for key, val in data.items():
            data[prefix + key] = data.pop(key)

        return data


class WindMoverFunctionalTests(FunctionalTestBase, WindMoverFixtures):

    def create_model(self):
        return self.testapp.post('/model/create', OrderedDict([
            ('confirm_new', True)
        ]))

    def test_create_wind_mover_with_multiple_time_series(self):
        data = self.create_wind_mover_data()
        hour = 1

        for idx in range(5):
            data.update(
                self.create_time_series_data(idx, hour=str(hour + idx)))

        self.create_model()
        resp = self.testapp.post('/model/mover/wind', data)

        json_resp = json.loads(resp.body)

        self.assertEqual(json_resp['form_html'], None)
        self.assertEqual(json_resp['type'], 'mover')
        self.assertIsNotNone(json_resp['id'])

    def test_create_wind_mover_with_one_time_series(self):
        data = self.create_wind_mover_data()
        data.update(
            self.create_time_series_data(0))

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

    def test_create_wind_mover(self):
        self.config.add_route('create_wind_mover', '/mover')

        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_wind_mover_data()
        data.update(self.create_time_series_data(1))

        request.POST = MultiDict(data)
        resp = create_wind_mover(request)

        self.assertEqual(model.movers[0].id, resp['id'])
