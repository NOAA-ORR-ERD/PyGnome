from webgnome.form_view import FormViewBase

from base import UnitTestBase


class TestObject(object):
    id = 1


class TestObject2(object):
    id = 1


class TestFormView(FormViewBase):
    wrapped_class = TestObject


class FormViewTests(UnitTestBase):
    def test_form_view_gets_form_id_for_object_class(self):
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(FormViewBase.get_form_id(TestObject), 'TestObject_create')

    def test_form_view_gets_form_id_for_object_instance(self):
        obj = TestObject()
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(
            FormViewBase.get_form_id(obj), 'TestObject_update_1')

        obj.id = 2

        self.assertEqual(
            FormViewBase.get_form_id(obj), 'TestObject_update_2')

    def test_form_view_ignores_objects_of_wrong_type(self):
        obj = TestObject2()
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(FormViewBase.get_form_id(obj), None)

    def test_form_view_uses_form_name_if_given(self):
        obj = TestObject()
        request = self.get_request()
        TestFormView(request)

        self.assertEqual(
            FormViewBase.get_form_id(obj, 'delete'), 'TestObject_delete_1')

    def test_form_view_requires_wrapped_class_attr(self):
        def fail():
            # Trying to define this class will fail.
            class TestFormView2(FormViewBase):
                pass

        self.assertRaises(AttributeError, fail)
