import json
import datetime
import numpy

from collections import OrderedDict
from webob.multidict import MultiDict

from base import FunctionalTestBase, UnitTestBase
from webgnome.model_manager import ModelManager
from webgnome.views.spills import (
    create_point_release_spill,
    update_point_release_spill
)


class Fixtures(object):
    def create_point_release_spill_data(self, **kwargs):
        data = OrderedDict([
            ('name', 'Point Release Spill'),
            ('is_active', 'y'),
            ('date', '01/01/2012'),
            ('hour', 1),
            ('minute', 10),
            ('start_position_x', 10),
            ('start_position_y', 20),
            ('start_position_z', 30),
            ('num_LEs', 1000),
            ('windage_min', 0.1),
            ('windage_max', 0.4),
            ('windage_persist', 900),
            ('is_uncertain', True),
            ('is_active', True)
        ])

        data.update(**kwargs)
        return data


class PointReleaseSpillFunctionalTests(FunctionalTestBase, Fixtures):

    def create_model(self):
        return self.testapp.post('/model/create', OrderedDict([
            ('confirm_new', True)
        ]))

    def test_create_wind_mover_with_multiple_time_series(self):
        data = self.create_point_release_spill_data()
        self.create_model()
        resp = self.testapp.post('/model/spill/point_release', data)

        json_resp = json.loads(resp.body)

        self.assertEqual(json_resp['form_html'], None)
        self.assertEqual(json_resp['type'], 'spill')
        self.assertIsNotNone(json_resp['id'])

    def test_create_wind_mover_with_one_time_series(self):
        data = self.create_point_release_spill_data()

        self.create_model()
        resp = self.testapp.post('/model/spill/point_release', data)

        json_resp = json.loads(resp.body)

        self.assertEqual(json_resp['form_html'], None)
        self.assertEqual(json_resp['type'], 'spill')
        self.assertIsNotNone(json_resp['id'])


class PointReleaseSpillUnitTests(UnitTestBase, Fixtures):
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
        self.config.add_route('create_point_release_spill', '/spill')

        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_point_release_spill_data()

        request.POST = MultiDict(data)
        resp = create_point_release_spill(request)


        self.assertTrue(resp['id'])

        spill = model.get_spill(resp['id'])

        self.assertEqual(spill.name, 'Point Release Spill')
        self.assertEqual(
            spill.release_time, datetime.datetime(2012, 1, 1, 1, 10))
        self.assertEqual(spill.start_position, (10.0, 20.0, 30.0))
        self.assertTrue(spill.is_active)
        self.assertEqual(spill.num_LEs, 1000)
        self.assertEqual(spill.windage_range, (0.1, 0.4))
        self.assertEqual(spill.windage_persist, 900)

    def test_update_point_release_spill_get(self):
        self.config.add_route('create_point_release_spill', '/spill')
        self.config.add_route('update_point_release_spill', '/spill/{id}')

        # Create the wind spill
        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_point_release_spill_data()
        request.POST = MultiDict(data)
        resp = create_point_release_spill(request)
        spill = model.get_spill(resp['id'])

        # The update wind spill form should have timeseries data.
        request.method = 'GET'
        request.POST = MultiDict()
        request.matchdict = MultiDict({'id': resp['id']})
        resp = update_point_release_spill(request)
        form_html = resp['form_html']

        self.assertIn('name="windage_max" type="text" value="0.4"', form_html)
        self.assertIn(
            'name="start_position_x" type="text" value="10.0"', form_html)
        self.assertIn(
            'name="start_position_y" type="text" value="20.0"', form_html)
        self.assertIn(
            'name="start_position_z" type="text" value="30.0"', form_html)
        self.assertIn('name="date" type="text" value="12/12/2012"', form_html)


    def test_update_point_release_spill_post(self):
        self.config.add_route('create_point_release_spill', '/spill')
        self.config.add_route('update_point_release_spill', '/spill/{id}')

        # Create the wind spill
        model, request = self.get_model_request()
        request.method = 'POST'
        data = self.create_point_release_spill_data()
        request.POST = MultiDict(data)
        resp = create_point_release_spill(request)
        spill = model.get_spill(resp['id'])

        # Update the wind spill with a new time series
        data = self.create_point_release_spill_data(
            name='Changed Spill Name',
            is_active='y',
            date='02/03/2012',
            hour=4,
            minute=30,
            start_position_x=30,
            start_position_y=40,
            start_position_z=50,
            num_LEs=400,
            windage_min=0.2,
            windage_max=0.5,
            windage_persist=500,
        )
        request.method = 'POST'
        request.POST = MultiDict(data)
        request.matchdict = MultiDict({'id': spill.id})
        resp = update_point_release_spill(request)

        self.assertEqual(resp['message']['type'], 'success')
        self.assertEqual(resp['id'], spill.id)

        spill = model.get_spill(spill.id)

        self.assertEqual(spill.name, 'Changed Spill Name')
        self.assertTrue(spill.is_active)
        self.assertEqual(
            spill.release_time, datetime.datetime(2012, 2, 3, 4, 30))
        self.assertEqual(spill.num_LEs, 400)
        self.assertEqual(spill.windage_range, (0.2, 0.5))
        self.assertEqual(spill.windage_persist, 500)
        self.assertEqual(spill.start_position, (30, 40, 50))
