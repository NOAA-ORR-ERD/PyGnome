import webgnome.util as util
import datetime

from webgnome.model_manager import ModelManager
from base import UnitTestBase


class JsonRequireModelTests(UnitTestBase):
    def make_request(self):
        request = self.get_request()
        request.registry.settings.model_session_key = \
            self.settings['model_session_key']
        request.registry.settings.Model = ModelManager()
        request.context = self.get_resource()
        return request

    def assert_is_missing_model_error(self, response):
        self.assertEqual(type(response), dict)
        self.assertTrue(response['error'])

        message = response['message']

        self.assertEqual(message['type'], 'error')
        self.assertEqual(message['text'], 'That model is no longer available.')

    def test_decorator_works_on_methods(self):
        class MockView(object):
            def __init__(self, request):
                self.request = request

            @util.json_require_model
            def method(self):
                return {}

        request = self.make_request()
        response = MockView(request).method()
        self.assert_is_missing_model_error(response)


    def test_decorator_works_on_function(self):
        @util.json_require_model
        def mock_view(response):
            return {}

        request = self.make_request()
        response = mock_view(request)
        self.assert_is_missing_model_error(response)


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

