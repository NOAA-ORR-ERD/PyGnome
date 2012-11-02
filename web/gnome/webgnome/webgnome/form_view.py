"""
form_view.py: A class-based view for form views that wrap other classes.
"""
from collections import defaultdict


class FormView(object):
    """
    A view that provides form routes for objects of a given class.

    The ``wrapped_class`` field is the class the view wraps.

    The view may contain any number of routes for instances of
    ``wrapped_class``, with subclasses providing the strategy for determining
    the correct route for a given instance via :meth:`_get_route_for_object`.

    The :meth:`get_route_for_object` class method is the public interface used
    to find a route for a given object if one is known by any :class`FormView`
    classes declared at runtime.
    """
    _registry = defaultdict(list)
    wrapped_class = None

    def __init__(self, request):
        """
        Add this object to the internal registry using the class of
        ``wrapped_class`` as the key.
        """
        if self.wrapped_class is None:
            raise AttributeError("A wrapped_class field is required.")

        self._registry[self.wrapped_class].append(self)
        self.request = request

    def _get_route_for_object(self, obj):
        """
        Return the route name most appropriate for ``obj``.
        """
        raise NotImplementedError

    @classmethod
    def get_route_for_object(cls, obj):
        """
        Iterate over the internal registry of all :class:`FormView` instances
        and ask each one for a route for the object, ending if a route is found.
        """
        form_views = cls._registry.get(obj.__class__, [])

        for view in form_views:
            if type(obj) == view.wrapped_class:
                route = view._get_route_for_object(obj)

                if route:
                    return route
