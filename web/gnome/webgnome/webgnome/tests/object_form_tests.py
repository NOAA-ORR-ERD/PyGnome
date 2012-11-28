from webgnome.forms.object_form import ObjectForm, get_object_form_cls
from base import UnitTestBase


class TestObject(object):
    id = 1


class TestObject2(object):
    id = 1


class TestObjectForm(ObjectForm):
    wrapped_class = TestObject


class ObjectFormTests(UnitTestBase):
    def test_get_object_form_returns_form_class_for_wrapped_class(self):
        TestObjectForm()
        self.assertEqual(get_object_form_cls(TestObject), TestObjectForm)

    def test_get_object_form_returns_form_class_for_wrapped_class_instance(self):
        TestObjectForm()
        obj = TestObject()
        self.assertEqual(get_object_form_cls(obj), TestObjectForm)

    def test_get_object_form_returns_none_for_objects_without_an_object_form(self):
        obj = TestObject2()
        self.assertEqual(get_object_form_cls(obj), None)

    def test_get_id_returns_form_id_for_wrapped_class_by_default(self):
        TestObjectForm()
        form_id = get_object_form_cls(TestObject).get_id()
        self.assertEqual(form_id, 'TestObjectForm')

    def test_get_id_returns_id_for_object_instance(self):
        obj = TestObject()
        form_id = get_object_form_cls(obj).get_id(obj)
        self.assertEqual(form_id, 'TestObjectForm_1')

        obj.id = 2
        form_id = get_object_form_cls(obj).get_id(obj)
        self.assertEqual(form_id, 'TestObjectForm_2')

    def test_object_form_requires_wrapped_class_attr(self):
        def fail():
            # Trying to define this class will fail.
            class BadObjectForm(ObjectForm):
                pass

        self.assertRaises(AttributeError, fail)
