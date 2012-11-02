from webgnome.form_view import FormView

from base import UnitTestBase


class TestObject(object):
    x = 1


class TestObject2(object):
    y = 1


class TestFormView(FormView):
    wrapped_class = TestObject

    route_1 = 'get_form_1'
    route_2 = 'get_form_2'

    def _get_route_for_object(self, obj):
        if obj.x == 1:
            return self.route_1
        else:
            return self.route_2


class TestFormView2(FormView):
    pass


class FormViewTests(UnitTestBase):
    def test_form_view_gets_route_for_object(self):
        obj = TestObject()
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(
            FormView.get_route_for_object(obj), TestFormView.route_1)

        obj.x = 2

        self.assertEqual(
            FormView.get_route_for_object(obj), TestFormView.route_2)

    def test_form_view_ignores_objects_of_wrong_type(self):
        obj = TestObject2()
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(FormView.get_route_for_object(obj), None)

    def test_form_view_requires_wrapped_class(self):
        request = self.get_request()
        self.assertRaises(AttributeError, TestFormView2, request)
