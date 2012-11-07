"""
object_form.py: A :class:`wtforms.Form` subclass that wraps an object.
"""
from wtforms.form import FormMeta

from webgnome import util
from base import AutoIdForm


_object_form_classes = {}


def get_object_form(obj):
    obj_class = util.get_obj_class(obj)
    return _object_form_classes.get(obj_class, None)


class ObjectFormMetaclass(FormMeta):
    """
    A metaclass that adds its subclasses to ``object_form_classes``.

    Subclasses must define a ``wrapped_class`` field that is used as the key
    in ``object_form_classes``, with the subclass as the value.

    The intention is to make it possible to look up a form class for an object
    at runtime.
    """
    def __new__(mcs, name, bases, dct):
        if 'wrapped_class' not in dct:
            raise AttributeError("A wrapped_class field is required.")

        wrapped_class = dct['wrapped_class']

        if wrapped_class in _object_form_classes:
            raise RuntimeError(
                "Form view already defined for %s" % wrapped_class)

        instance = super(ObjectFormMetaclass, mcs).__new__(
            mcs, name, bases, dct)

        if wrapped_class:
            # Register this object form class as handling ``wrapped_class``.
            _object_form_classes[wrapped_class] = instance

        return instance


class ObjectForm(AutoIdForm):
    """
    A form that "wraps" a class.

    An :class:`ObjectForm` subclass that may be looked up by ``wrapped_class``
    or an instance of ``wrapped_class`` at runtime.
    """
    wrapped_class = None
    __metaclass__ = ObjectFormMetaclass

