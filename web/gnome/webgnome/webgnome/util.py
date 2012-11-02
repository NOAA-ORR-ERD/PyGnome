"""
util.py: Utility function for the webgnome package.
"""
import datetime
import inspect
from functools import wraps


def make_message(type, text):
    """
    Create a dictionary suitable to be returned in a JSON response as a
    "message" sent to the JavaScript client.

    The client looks for "message" objects included in JSON responses that the
    server sends back on successful form submits and if one is present, and has
    a ``type`` field and ``text`` field, it will display the message to the
    user.
    """
    return dict(type=type, text=text)


def encode_json_date(obj):
    """
    Render a :class:`datetime.datetime` or :class:`datetime.date` object using
    the :meth:`datetime.isoformat` function, so it can be properly serialized to
    JSON notation.
    """
    if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date):
        return obj.isoformat()


def json_encoder(obj):
    """
    A custom JSON encoder that handles :class:`datetime.datetime` and
    :class:`datetime.date` values, with a fallback to using the :func:`str`
    representation of an object.
    """
    date_str = encode_json_date(obj)

    if date_str:
        return date_str
    else:
        return str(obj)


def json_date_adapter(obj, request):
    """
    A wrapper around :func:`json_date_encoder` so that it may be used in a
    custom JSON adapter for a Pyramid renderer.
    """
    return encode_json_date(obj)


def get_model_from_request(request):
    """
    Return a :class:`gnome.model.Model` if the user has a session key that
    matches the ID of a running model.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)

    try:
        model = settings.Model.get(model_id)
    except settings.Model.DoesNotExist:
        model = None

    return model


MISSING_MODEL_ERROR = {
    'error': True,
    'message': make_message('error', 'That model is no longer available.')
}


def json_require_model(f):
    """
    Wrap a JSON view in a precondition that checks if the user has a valid
    ``model_id`` in his or her session and fails if not.

    If the key is missing or no model is found for that key, return a JSON
    object with a ``message`` object describing the error.

    This decorator works on functions and methods. It returns a method decorator
    if the first argument to the function is ``self``. Otherwise, it returns a
    function decorator.
    """
    args = inspect.getargspec(f)

    if args and args.args[0] == 'self':
        @wraps(f)
        def inner_method(self, *args, **kwargs):
            model = get_model_from_request(self.request)
            if model is None:
                return MISSING_MODEL_ERROR
            return f(self, model, *args, **kwargs)
        wrapper = inner_method
    else:
        @wraps(f)
        def inner_fn(request, *args, **kwargs):
            model = get_model_from_request(request)
            if model is None:
                return MISSING_MODEL_ERROR
            return f(request, model, *args, **kwargs)
        wrapper = inner_fn
    return wrapper
