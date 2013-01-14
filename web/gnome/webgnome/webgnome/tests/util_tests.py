import webgnome.util as util
import datetime

from webgnome.model_manager import ModelManager, WebModel
from base import UnitTestBase


class JsonRequireModelTests(UnitTestBase):
    def make_request(self):
        request = self.get_request()
        request.registry.settings.model_session_key = \
            self.settings['model_session_key']
        request.registry.settings.model_images_dir = \
            self.settings['model_images_dir']
        request.registry.settings.Model = ModelManager()
        request.context = self.get_resource()
        return request

    def test_function_decorator_creates_model_if_one_doesnt_exist(self):
        class MockView(object):
            def __init__(self, request):
                self.request = request

            @util.require_model
            def method(self, model):
                return model

        request = self.make_request()
        response = MockView(request).method()
        self.assertEqual(type(response), WebModel)
        self.assertTrue(response.id)

    def test_method_decorator_creates_model_if_one_doesnt_exist(self):
        @util.require_model
        def mock_view(response, model):
            return model

        request = self.make_request()
        response = mock_view(request)
        self.assertEqual(type(response), WebModel)
        self.assertTrue(response.id)

    def test_function_decorator_uses_existing_model_if_one_exists(self):
        @util.require_model
        def mock_view(response, model):
            return model

        request = self.make_request()
        model = request.registry.settings.Model.create(
            model_images_dir=self.settings['model_images_dir'])
        request.session[self.settings['model_session_key']] = model.id
        response = mock_view(request)
        self.assertEqual(response, model)
        self.assertTrue(response.id, model.id)

    def test_method_decorator_uses_existing_model_if_one_exists(self):
        class MockView(object):
            def __init__(self, request):
                self.request = request

            @util.require_model
            def method(self, model):
                return model

        request = self.make_request()
        model = request.registry.settings.Model.create(
            model_images_dir=self.settings['model_images_dir'])
        request.session[self.settings['model_session_key']] = model.id
        response = MockView(request).method()
        self.assertEqual(response, model)
        self.assertTrue(response.id, model.id)


class MakeMessageTests(UnitTestBase):
    def test_make_message_returns_correct_dict(self):
        message = util.make_message('error', 'Error text')
        self.assertEqual(type(message), dict)
        self.assertEqual(message['type'], 'error')
        self.assertEqual(message['text'], 'Error text')


class EncodeJsonDateTests(UnitTestBase):
    def test_can_encode_datetime(self):
        date_time = datetime.datetime(year=2012, month=1, day=1, hour=1,
                                      minute=1, second=1)
        encoded = util.encode_json_date(date_time)
        self.assertEqual(encoded, "2012-01-01T01:01:01")

    def test_can_encode_date(self):
        date = datetime.date(year=2012, month=1, day=1)
        encoded = util.encode_json_date(date)
        self.assertEqual(encoded, "2012-01-01")

    def test_return_none_if_not_date_or_datetime(self):
        not_a_date = "hello!"
        self.assertEqual(util.encode_json_date(not_a_date), None)


class JsonEncoderTests(UnitTestBase):
    def test_can_encode_datetime(self):
        date_time = datetime.datetime(year=2012, month=1, day=1, hour=1,
            minute=1, second=1)
        encoded = util.json_encoder(date_time)
        self.assertEqual(encoded, "2012-01-01T01:01:01")

    def test_can_encode_date(self):
        date = datetime.date(year=2012, month=1, day=1)
        encoded = util.json_encoder(date)
        self.assertEqual(encoded, "2012-01-01")

    def test_can_encode_objects_with_str_method(self):
        class CustomObject(object):
            def __str__(self):
                return "Wahoo"

        obj = CustomObject()
        self.assertEqual(util.json_encoder(obj), "Wahoo")

