"""
form_view.py: A class-based view for form views that wrap other classes.
"""


_form_views = {}



def get_obj_class(obj):
    return obj if type(obj) == type else obj.__class__


def get_form_view(obj):
    obj_class = get_obj_class(obj)
    return _form_views.get(obj_class, None)


def get_form_container_id(obj, form_name=None):
    """
    Get the container ID for ``obj`` if a :class:`FormViewBase` subclass
    was declared that wraps ``obj``.
    """
    form_view = get_form_view(obj)
    obj_class = get_obj_class(obj)

    if form_view and obj_class == form_view.wrapped_class:
        form_id = form_view._get_form_container_id(obj, form_name)

        if form_id:
            return form_id


class FormViewMetaclass(type):
    def __new__(mcs, name, bases, dct):
        if 'wrapped_class' not in dct:
            raise AttributeError("A wrapped_class field is required.")

        wrapped_class = dct['wrapped_class']

        if wrapped_class in _form_views:
            raise RuntimeError(
                "Form view already defined for %s" % wrapped_class)

        instance = super(FormViewMetaclass, mcs).__new__(mcs, name, bases, dct)

        if wrapped_class:
            # Add this object to the dict of form view classes.
            _form_views[wrapped_class] = instance

        return instance


class FormViewBase(object):
    """
    A Pyramid class-based view that provides form routes for objects of a
    "wrapped" class.

    :class:`FormView` provides a class method that will return a form ID for an
    object if a :class:`FormView` subclass is declared that wraps the object's
    class.

    The ``wrapped_class`` field is the class the view wraps. The view may
    contain any number of Pyramid routes for objects of that class.

    :meth:`webgnome.form_view.FormViewBase.get_form_id` is the public interface
    used to find a form ID for a given object. Subclasses of
    :class:`FormView: provide the strategy for determining the correct ID for
    the object by implementing :meth:`FormViewBase._get_form_id_for_object`.
    """
    wrapped_class = None
    __metaclass__ = FormViewMetaclass

    def __init__(self, request):
        self.request = request

    @classmethod
    def _get_form_id(cls, append_with):
        return '%s_%s' % (cls.wrapped_class.__name__, append_with)

    @classmethod
    def _get_form_container_id(cls, obj, form_name=None):
        """
        Return a unique form container ID for ``obj``.

        If ``obj`` has the class of ``self.wrapped_class`, then use "create"
        (or ``form_name`` if specified) as the form name.

        If ``obj`` is an instance of the class and has an ID, then use "edit"
        (or ``form_name`` if specified) as the form name appended by the
        object's ID, e.g., "object_1_edit" for an ``obj`` instance with ID 1.
        """
        if obj == cls.wrapped_class:
            form_name = form_name if form_name else 'create'
        elif obj.__class__ is cls.wrapped_class and \
                hasattr(obj, 'id') and obj.id:
            form_name = '%s_%s' % (
                form_name if form_name else 'update' , obj.id)

        return cls._get_form_id(form_name)

